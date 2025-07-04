{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Torch
import qualified Torch.NN as NN
import Torch.Optim (GD(..), Adam, LearningRate, mkAdam, adam)
import qualified Torch.Functional as F
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
import Torch.Train (update, showLoss, sumTensors)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.List (foldl')
import Control.Monad
import Control.Exception as Control.Exception
import Control.DeepSeq (deepseq, force, NFData(..))
import System.Mem (performGC)
import GHC.Generics (Generic)
import Data.Maybe (mapMaybe)
import Text.Printf (printf)
import Debug.Trace (trace)
import Prelude hiding (div, sqrt, max, any, mod, abs, tanh)  -- Hide ambiguous functions from Prelude
import qualified Prelude
import Data.Char (toLower)
-- Data types
type WordEmbedding = Tensor
type WordPair = (String, String)
type Label = Bool
type WordEmbeddingMap = Map.Map String Tensor

-- Simplified MLP using hasktorch-tools utilities
data MLPSpec = MLPSpec {
    inputSize :: !Int,
    hiddenSize1 :: !Int,
    hiddenSize2 :: !Int,
    outputSize :: !Int,
    dropoutRate :: !Double
} deriving (Show)

data MLP = MLP {
    linear1 :: !Linear,
    linear2 :: !Linear,
    linear3 :: !Linear,
    dropoutRate :: !Double
} deriving (Generic, Show)

instance Parameterized MLP
instance HasForward MLP Tensor Tensor where
    forward model input = do
        let !x1 = Torch.tanh $ linear (linear1 model) input
        !x1_dropped <- F.dropout (dropoutRate model) True x1
        let !x2 = Torch.tanh $ linear (linear2 model) x1_dropped
        !x2_dropped <- F.dropout (dropoutRate model) True x2
        let !output = sigmoid $ linear (linear3 model) x2_dropped
        return output

-- Custom NFData instance since Linear doesn't have NFData
instance NFData MLP where
    rnf (MLP l1 l2 l3 dr) = l1 `seq` l2 `seq` l3 `seq` dr `seq` ()

-- Custom NFData instance for Tensor (if not already available)
instance NFData Tensor where
    rnf tensor = tensor `seq` ()

mkMLP :: MLPSpec -> IO MLP
mkMLP spec = do
    -- Use proper Xavier/He initialization for better stability
    linear1 <- sample $ NN.LinearSpec (inputSize spec) (hiddenSize1 spec)
    linear2 <- sample $ NN.LinearSpec (hiddenSize1 spec) (hiddenSize2 spec)
    linear3 <- sample $ NN.LinearSpec (hiddenSize2 spec) (outputSize spec)

    let !model = MLP linear1 linear2 linear3 (dropoutRate spec)
    return model

-- Simplified forward pass using HasForward typeclass
mlpForward :: MLP -> Tensor -> Bool -> Double -> IO Tensor
mlpForward model input isTraining _ = 
    if isTraining 
    then forward model input
    else do
        -- Evaluation mode - no dropout
        let !x1 = Torch.tanh $ linear (linear1 model) input
        let !x2 = Torch.tanh $ linear (linear2 model) x1
        let !output = sigmoid $ linear (linear3 model) x2
        return output

-- Enhanced features with robust normalization and NaN protection
prepareEnhancedFeatures :: WordEmbedding -> WordEmbedding -> Tensor
prepareEnhancedFeatures emb1 emb2 =
    let -- Ensure embeddings are Float type and check for NaN/Inf
        !emb1_float = toType Float emb1
        !emb2_float = toType Float emb2
        
        -- Check for NaN/Inf in input embeddings
        hasNaN1 = hasNaNOrInf emb1_float
        hasNaN2 = hasNaNOrInf emb2_float

        -- L2 normalize embeddings with more robust handling
        !norm1_sq = asValue (sumAll (mul emb1_float emb1_float)) :: Float
        !norm2_sq = asValue (sumAll (mul emb2_float emb2_float)) :: Float
        !norm1 = Prelude.max 1e-6 (Prelude.sqrt norm1_sq)  -- Increase minimum norm
        !norm2 = Prelude.max 1e-6 (Prelude.sqrt norm2_sq)
        !emb1_norm = if hasNaN1 then zeros' [100] else Torch.div emb1_float (full' [] norm1)
        !emb2_norm = if hasNaN2 then zeros' [100] else Torch.div emb2_float (full' [] norm2)

        -- Basic features with clipping
        !concatenated = cat (Dim 0) [emb1_norm, emb2_norm]
        !difference = clamp (-10.0) 10.0 $ sub emb1_norm emb2_norm  -- Clip to prevent extreme values
        !elementwise_product = clamp (-10.0) 10.0 $ mul emb1_norm emb2_norm

        -- Additional similarity features with safer computation
        !dot_product = sumAll $ mul emb1_norm emb2_norm
        !cosine_sim = reshape [1] $ clamp (-0.999) 0.999 dot_product  -- More conservative bounds
        !diff_sq_sum = sumAll $ mul difference difference
        !euclidean_dist = reshape [1] $ clamp 0.0 3.0 $ Torch.sqrt $ add diff_sq_sum (full' [] (1e-8 :: Float))

        -- Combine all features without final normalization to avoid division issues
        !combined = cat (Dim 0) [concatenated, difference, elementwise_product, cosine_sim, euclidean_dist]
        
        -- Simple clipping instead of normalization to avoid NaN
        !result = clamp (-5.0) 5.0 combined
    in result `seq` result

-- Improved GloVe embeddings loader with better error handling
loadGloVeEmbeddings :: FilePath -> Set.Set String -> IO WordEmbeddingMap
loadGloVeEmbeddings filepath requiredWords = do
    putStrLn $ "Loading GloVe embeddings for " ++ show (Set.size requiredWords) ++ " required words..."
    putStrLn $ "Reading from: " ++ filepath

    result <- Control.Exception.catch
        (do content <- readFile filepath
            let linesContent = lines content
            putStrLn $ "File contains " ++ show (length linesContent) ++ " lines"

            let embeddings = parseGloVeLines linesContent requiredWords Map.empty 0
            putStrLn $ "Successfully loaded " ++ show (Map.size embeddings) ++ " embeddings"

            -- Check if we got most of the required words
            let foundWords = Set.fromList (Map.keys embeddings)
                missingWords = Set.difference requiredWords foundWords
                coveragePercent = (fromIntegral (Set.size foundWords) * 100.0) / fromIntegral (Set.size requiredWords) :: Float

            putStrLn $ printf "Coverage: %.1f%% (%d/%d words found)"
                       coveragePercent (Set.size foundWords) (Set.size requiredWords)

            when (Set.size missingWords > 0) $ do
                putStrLn $ "Missing " ++ show (Set.size missingWords) ++ " words"
                when (Set.size missingWords <= 10) $ do
                    putStrLn $ "Missing words: " ++ show (Set.toList missingWords)

            performGC
            return embeddings)
        (\e -> do
            putStrLn $ "Exception while loading GloVe file: " ++ show (e :: Control.Exception.SomeException)
            return Map.empty)
    return result

parseGloVeLines :: [String] -> Set.Set String -> WordEmbeddingMap -> Int -> WordEmbeddingMap
parseGloVeLines [] _ acc _ = acc
parseGloVeLines (line:rest) requiredWords acc processed =
    let newAcc = case words line of
                   (word:vectorStrs)
                     | Set.member word requiredWords && not (null vectorStrs) ->
                         case mapM readFloat vectorStrs of
                           Just floats ->
                             if length floats == 100  -- Expecting 100-dimensional GloVe
                             then Map.insert word (asTensor floats) acc
                             else if processed < 10  -- Show first few dimension mismatches
                                  then Debug.Trace.trace ("Warning: word '" ++ word ++ "' has " ++ show (length floats) ++ " dimensions, expected 100") acc
                                  else acc
                           Nothing ->
                             if processed < 10  -- Show first few parsing errors
                             then Debug.Trace.trace ("Warning: failed to parse numbers for word '" ++ word ++ "'") acc
                             else acc
                     | otherwise -> acc
                   [] -> acc  -- Empty line
                   [_] -> acc  -- Line with only word, no vectors
        newProcessed = processed + 1
    in if newProcessed `Prelude.mod` 100000 == 0  -- Less frequent progress updates
       then Debug.Trace.trace ("Processed " ++ show newProcessed ++ " lines, found " ++ show (Map.size newAcc) ++ " required words") $
            parseGloVeLines rest requiredWords newAcc newProcessed
       else parseGloVeLines rest requiredWords newAcc newProcessed
  where
    readFloat :: String -> Maybe Float
    readFloat s = case reads s of
                    [(f, "")] | not (isNaN f || isInfinite f) -> Just f  -- Validate the float
                    _ -> Nothing

-- Fallback: Load all GloVe embeddings
loadAllGloVeEmbeddings :: FilePath -> IO WordEmbeddingMap
loadAllGloVeEmbeddings filepath = do
    putStrLn "Loading all GloVe embeddings..."
    result <- Control.Exception.catch
        (do content <- readFile filepath
            let embeddings = parseAllGloVeLines (lines content) Map.empty 0
            putStrLn $ "Loaded " ++ show (Map.size embeddings) ++ " total embeddings"
            performGC
            return embeddings)
        (\e -> do
            putStrLn $ "Exception while loading GloVe file: " ++ show (e :: Control.Exception.SomeException)
            return Map.empty)
    return result

parseAllGloVeLines :: [String] -> WordEmbeddingMap -> Int -> WordEmbeddingMap
parseAllGloVeLines [] acc _ = acc
parseAllGloVeLines (line:rest) acc processed =
    let newAcc = case words line of
                   (word:vectorStrs) ->
                       case mapM readFloat vectorStrs of
                         Just floats ->
                           if length floats == 100
                           then Map.insert word (asTensor floats) acc
                           else acc
                         Nothing -> acc
                   _ -> acc
        newProcessed = processed + 1
    in if newProcessed `Prelude.mod` 50000 == 0
       then Debug.Trace.trace ("Processed " ++ show newProcessed ++ " lines, loaded " ++ show (Map.size newAcc) ++ " embeddings") $
            parseAllGloVeLines rest newAcc newProcessed
       else parseAllGloVeLines rest newAcc newProcessed
  where
    readFloat :: String -> Maybe Float
    readFloat s = case reads s of
                    [(f, "")] -> Just f
                    _ -> Nothing

-- Load dataset from CSV
loadDataset :: FilePath -> IO [(WordPair, Label)]
loadDataset filepath = do
    putStrLn $ "Loading dataset from: " ++ filepath
    content <- readFile filepath
    let allLines = lines content
        (validLines, invalidCount) = foldl' processLine ([], 0) (zip [1..] allLines)
        !result = reverse validLines

    when (invalidCount > 0) $
        putStrLn $ "Warning: Skipped " ++ show invalidCount ++ " invalid lines"

    putStrLn $ "Successfully loaded " ++ show (length result) ++ " examples"

    let positives = length $ filter snd result
        negatives = length result - positives
    putStrLn $ "Label distribution - Positive: " ++ show positives ++ ", Negative: " ++ show negatives

    result `deepseq` return result
  where
    processLine (acc, errCount) (lineNum, line) =
        case splitOn ',' line of
            [w1, w2, labelStr] ->
                case reads (trim labelStr) of
                    [(labelInt, "")] ->
                        let label = labelInt == (1 :: Int)
                            cleanW1 = trim $ map toLower  w1
                            cleanW2 = trim $ map toLower w2
                        in if null cleanW1 || null cleanW2
                           then (acc, errCount + 1)
                           else (((cleanW1, cleanW2), label) : acc, errCount)
                    _ -> (acc, errCount + 1)
            _ -> (acc, errCount + 1)

    trim :: String -> String
    trim = reverse . dropWhile isSpace . reverse . dropWhile isSpace
      where isSpace c = c `elem` [' ', '\t', '\r', '\n']

splitOn :: Char -> String -> [String]
splitOn delim = foldr f [[]]
  where
    f c (x:xs) | c == delim = [] : x : xs
               | otherwise  = (c:x) : xs
    f _ [] = []

-- Dataset preparation with validation and debugging
prepareDataset :: WordEmbeddingMap -> [(WordPair, Label)] -> [(Tensor, Tensor)]
prepareDataset embeddings dataset =
    let results = mapMaybe convert dataset
        totalAttempted = length dataset
        successful = length results
        missingWords = filter (\((w1, w2), _) -> isNothing (Map.lookup w1 embeddings) || isNothing (Map.lookup w2 embeddings)) dataset
    in Debug.Trace.trace ("Dataset conversion: " ++ show successful ++ "/" ++ show totalAttempted ++ " successful") $
       Debug.Trace.trace ("Missing word pairs: " ++ show (length missingWords)) $
       if length missingWords <= 10 
       then Debug.Trace.trace ("First few missing: " ++ show (Prelude.take 5 missingWords)) results
       else results
  where
    convert ((w1, w2), label) = do
        e1 <- Map.lookup w1 embeddings
        e2 <- Map.lookup w2 embeddings
        let !features = prepareEnhancedFeatures e1 e2
            !target = asTensor ([if label then 1.0 else 0.0 :: Float] :: [Float])
        -- Check for NaN in features
        if hasNaNOrInf features
        then Nothing  -- Skip examples with NaN features
        else return (features, target)
    
    isNothing Nothing = True
    isNothing (Just _) = False

splitDataset :: [(Tensor, Tensor)] -> Float -> ([(Tensor, Tensor)], [(Tensor, Tensor)])
splitDataset dataset ratio =
    let n = length dataset
        trainN = round $ ratio * fromIntegral n
    in splitAt trainN dataset

-- Simplified batch utilities using hasktorch-tools
makeBatches :: Int -> [(Tensor, Tensor)] -> [[(Tensor, Tensor)]]
makeBatches _ [] = []
makeBatches bs xs = let (!b, !rest) = splitAt bs xs in b : makeBatches bs rest

mergeBatch :: [(Tensor, Tensor)] -> (Tensor, Tensor)
mergeBatch batch =
    let !inputs = stack (Dim 0) (map fst batch)
        !targets = reshape [-1] $ stack (Dim 0) (map snd batch)
    in (inputs, targets)

-- Create data pipeline using hasktorch-tools
createDataPipeline :: [(Tensor, Tensor)] -> Int -> [(Tensor, Tensor)]
createDataPipeline dataset batchSize = 
    let batches = makeBatches batchSize dataset
    in map mergeBatch batches

-- Simplified training using hasktorch-tools utilities
trainModel :: MLP -> [(Tensor, Tensor)] -> Int -> Float -> Int -> Double -> IO MLP
trainModel model dataset epochs lr batchSize dropRate = do
    putStrLn $ "Training with " ++ show (length dataset) ++ " examples for " ++ show epochs ++ " epochs"
    putStrLn $ "Learning rate: " ++ show lr ++ ", Batch size: " ++ show batchSize ++ ", Dropout: " ++ show dropRate

    let params = flattenParameters model
        adamOptimizer = mkAdam 0 0.9 0.999 params

    (finalModel, _) <- foldM (trainEpoch lr batchSize dataset) (model, adamOptimizer) [1..epochs]
    return finalModel

trainEpoch :: Float -> Int -> [(Tensor, Tensor)] -> (MLP, Adam) -> Int -> IO (MLP, Adam)
trainEpoch lr batchSize dataset (currentModel, adamOptim) epoch = do
    let !batches = makeBatches batchSize dataset
    (updatedModel, updatedOptim, totalLoss, batchCount) <-
        foldM (trainBatch lr) (currentModel, adamOptim, 0.0, 0) batches

    performGC

    let !avgLoss = if batchCount > 0 then totalLoss / fromIntegral batchCount else 0.0
    printf "Epoch %d - Avg loss: %.6f (batches: %d)\n" epoch avgLoss batchCount

    when (epoch <= 5) $ do
        printf "  Learning rate: %.6f\n" lr
        printf "  Avg loss: %.6f, Batches processed: %d\n" avgLoss batchCount
        when (isNaN avgLoss || isInfinite avgLoss || avgLoss > 10.0) $
            putStrLn "  WARNING: Loss is unstable!"

    if isNaN avgLoss || isInfinite avgLoss then do
        putStrLn "ERROR: Loss is NaN or Infinite! Training may have diverged."
        return (currentModel, adamOptim)
    else
        return (updatedModel, updatedOptim)

-- Simplified training batch using runStep utility
trainBatch :: Float -> (MLP, Adam, Float, Int) -> [(Tensor, Tensor)] -> IO (MLP, Adam, Float, Int)
trainBatch lr (model', adamOptim, accLoss, batchCount) batch = do
    let (!inputs, !targets) = mergeBatch batch
    
    -- Use runStep for cleaner training step
    let lossFunc = \m -> do
            !predictions <- forward m inputs
            let !clampedPredictions = clamp 1e-7 (1.0 - 1e-7) $ reshape [-1] predictions
                !clampedTargets = toType Float targets
            return $ F.binaryCrossEntropyLoss' clampedPredictions clampedTargets
    
    (!newModel, !newOptim, !lossTensor) <- runStep model' adamOptim lossFunc lr
    let !lossScalar = asValue lossTensor :: Float

    if isNaN lossScalar || isInfinite lossScalar || lossScalar > 100.0 then do
        putStrLn $ "WARNING: Bad loss in batch " ++ show (batchCount + 1) ++ ": " ++ show lossScalar
        return (model', adamOptim, accLoss, batchCount + 1)
    else do
        return (newModel, newOptim, accLoss + lossScalar, batchCount + 1)

-- Evaluation
predict :: MLP -> Tensor -> IO Bool
predict model input = do
    !p <- mlpForward model input False 0.0  -- Evaluation mode without dropout
    return $ (asValue p :: Float) > 0.5

predictProb :: MLP -> Tensor -> IO Float
predictProb model input = do
    !result <- mlpForward model input False 0.0
    return (asValue result :: Float)

evaluateModel :: MLP -> [(Tensor, Tensor)] -> [(WordPair, Label)] -> IO Float
evaluateModel model ds originalPairs = do
    results <- zipWithM evalSample ds originalPairs
    let !validResults = filter (not . isNaN . (\(_, _, prob, _) -> prob)) results
        !preds = map (\(_, _, _, correct) -> correct) validResults
        !correct = length $ filter id preds
        !total = length validResults

    if total == 0 then do
        putStrLn "ERROR: No valid predictions (all NaN)"
        return (0.0/0.0)
    else do
        let !acc = fromIntegral correct / fromIntegral total
        printf "Accuracy: %.2f%% (%d/%d valid samples)\n" (acc * 100) correct total

        putStrLn "\n=== Detailed Predictions (first 20 samples) ==="
        mapM_ showPrediction (Prelude.take 20 validResults)

        let incorrectPreds = filter (not . (\(_, _, _, correct) -> correct)) validResults
        when (not $ null incorrectPreds) $ do
            putStrLn "\n=== Incorrect Predictions (first 10) ==="
            mapM_ showPrediction (Prelude.take 10 incorrectPreds)

        when (length validResults < length ds) $
            putStrLn $ "Warning: " ++ show (length ds - length validResults) ++ " samples produced NaN predictions"
        return acc
  where
    evalSample (x, y) (wordPair, actualLabel) = do
        !prediction <- mlpForward model x False 0.0
        let !predictionFloat = asValue prediction :: Float
            !actual = (asValue y :: Float) > 0.5
            !predicted = predictionFloat > 0.5
            !isCorrect = predicted == actual
        return (wordPair, actualLabel, predictionFloat, isCorrect)

    showPrediction ((w1, w2), actualLabel, prob, isCorrect) =
        let predLabel = prob > 0.5
            status = if isCorrect then "✓" else "✗" :: String
            actualStr = if actualLabel then "hypernym" else "not hypernym" :: String
            predStr = if predLabel then "hypernym" else "not hypernym" :: String
        in printf "%s %s -> %s | Actual: %s, Predicted: %s (%.3f)\n"
                  status w1 w2 actualStr predStr prob

-- Interactive prediction
predictWordPair :: MLP -> WordEmbeddingMap -> String -> String -> IO ()
predictWordPair model embeddings word1 word2 = do
    case (Map.lookup word1 embeddings, Map.lookup word2 embeddings) of
        (Just emb1, Just emb2) -> do
            let !features = prepareEnhancedFeatures emb1 emb2
            !prob <- predictProb model features
            let !prediction = prob > 0.5
                !relationship = if prediction then "IS a hypernym of" else "is NOT a hypernym of" :: String
            printf "\nPrediction: '%s' %s '%s'\n" word1 relationship word2
            printf "Confidence: %.3f (%.1f%%)\n" prob (prob * 100)
        (Nothing, _) -> putStrLn $ "Error: Word '" ++ word1 ++ "' not found in embeddings"
        (_, Nothing) -> putStrLn $ "Error: Word '" ++ word2 ++ "' not found in embeddings"

extractRequiredWords :: [(WordPair, Label)] -> Set.Set String
extractRequiredWords dataset =
    let !words = concatMap (\((w1, w2), _) -> [w1, w2]) dataset
    in Set.fromList words

-- Helper function to check for NaN or infinite values in tensors
hasNaNOrInf :: Tensor -> Bool
hasNaNOrInf tensor =
    let nanTensor = isnan tensor
        nanCount = asValue (sumAll (toType Float nanTensor)) :: Float
        -- Check for infinite values by comparing with very large numbers
        absVals = Torch.abs tensor
        largeVals = gt absVals (full' [] (1e30 :: Float))
        infCount = asValue (sumAll (toType Float largeVals)) :: Float
    in nanCount > 0 || infCount > 0

-- MAIN
main :: IO ()
main = do
    putStrLn "=== Loading dataset first ==="
    !dataset <- loadDataset "train.csv"

    when (null dataset) $ do
        putStrLn "ERROR: No dataset loaded!"
        return ()

    putStrLn $ "Dataset loaded: " ++ show (length dataset) ++ " examples"

    let !requiredWords = extractRequiredWords dataset
    putStrLn $ "Found " ++ show (Set.size requiredWords) ++ " unique words"

    putStrLn "=== Loading filtered GloVe embeddings ==="
    !embeddings <- loadGloVeEmbeddings "glove.6B.100d.txt" requiredWords

    !finalEmbeddings <- if Map.size embeddings < Set.size requiredWords `Prelude.div` 2 then do
        putStrLn $ "Filtered loading got too few embeddings (" ++ show (Map.size embeddings) ++ "). Trying full load..."
        !fullEmbeddings <- loadAllGloVeEmbeddings "glove.6B.100d.txt"
        when (Map.null fullEmbeddings) $ do
            putStrLn "ERROR: Failed to load any embeddings!"
        return fullEmbeddings
    else do
        putStrLn "Filtered loading successful!"
        return embeddings

    when (Map.null finalEmbeddings) $ do
        putStrLn "ERROR: No embeddings available!"
        return ()

    putStrLn "=== Preparing dataset ==="
    let !tensorData = prepareDataset finalEmbeddings dataset
    putStrLn $ "Prepared " ++ show (length tensorData) ++ " tensor examples"

    when (null tensorData) $ do
        putStrLn "ERROR: No valid dataset examples after embedding lookup."
        putStrLn "Check that your GloVe file path is correct and contains the required words."
        return ()

    when (length tensorData < 100) $ do
        putStrLn $ "WARNING: Very few examples (" ++ show (length tensorData) ++ "). This may cause training issues."

    -- Check for NaN or infinite values in the prepared dataset
    let hasNaN = Prelude.any (\(x, y) -> hasNaNOrInf x || hasNaNOrInf y) (Prelude.take 10 tensorData)
    when hasNaN $ do
        putStrLn "WARNING: Found NaN or infinite values in prepared dataset!"

    putStrLn "Dataset preparation completed successfully."

    let (!train, !test) = splitDataset tensorData 0.8
        (!trainPairs, !testPairs) = splitAt (length train) dataset
        !embeddingDim = 100  -- GloVe 100d
        -- FIXED: Correct feature dimension calculation
        -- concat(200) + diff(100) + product(100) + cosine(1) + euclidean(1) = 402
        !featureDim = embeddingDim * 4 + 2
        !spec = MLPSpec featureDim 256 128 1 0.3  -- Added dropout

    putStrLn $ "Train: " ++ show (length train) ++ ", Test: " ++ show (length test)
    putStrLn $ "Feature dimension: " ++ show featureDim

    putStrLn "=== Creating model ==="
    !model <- mkMLP spec

    putStrLn "=== Training ==="
    -- FIXED: Much better hyperparameters to prevent NaN loss
    !trainedModel <- trainModel model train 20 0.0001 16 0.0  -- Lower LR (0.0001), smaller batch (16), no dropout

    putStrLn "=== Evaluation ==="
    !_ <- evaluateModel trainedModel test testPairs

    putStrLn "\n=== Interactive Predictions ==="
    putStrLn "Testing some example word pairs:"

    predictWordPair trainedModel finalEmbeddings "animal" "dog"
    predictWordPair trainedModel finalEmbeddings "vehicle" "car"
    predictWordPair trainedModel finalEmbeddings "food" "apple"
    predictWordPair trainedModel finalEmbeddings "dog" "animal"

    when (not $ null testPairs) $ do
        putStrLn "\nTesting pairs from the test dataset:"
        let samplePairs = Prelude.take 5 testPairs
        mapM_ (\((w1, w2), _) -> predictWordPair trainedModel finalEmbeddings w1 w2) samplePairs

    putStrLn "\n=== Final cleanup ==="
    performGC
    putStrLn "Done!"
