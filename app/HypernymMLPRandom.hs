{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Torch
import qualified Torch.NN as NN
import Torch.Optim (Adam, LearningRate, mkAdam, adam)
import qualified Torch.Functional as F
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
import System.Random (randomRIO)

-- Data types
type WordEmbedding = Tensor
type WordPair = (String, String)
type Label = Bool
type WordEmbeddingMap = Map.Map String Tensor

-- MLP with strict fields and dropout
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
    linear3 :: !Linear
} deriving (Generic, Show)

instance Parameterized MLP

-- Custom NFData instance since Linear doesn't have NFData
instance NFData MLP where
    rnf (MLP l1 l2 l3) = l1 `seq` l2 `seq` l3 `seq` ()

-- Custom NFData instance for Tensor (if not already available)
instance NFData Tensor where
    rnf tensor = tensor `seq` ()

mkMLP :: MLPSpec -> IO MLP
mkMLP spec = do
    -- Use proper Xavier/He initialization for better stability
    linear1 <- sample $ NN.LinearSpec (inputSize spec) (hiddenSize1 spec)
    linear2 <- sample $ NN.LinearSpec (hiddenSize1 spec) (hiddenSize2 spec)
    linear3 <- sample $ NN.LinearSpec (hiddenSize2 spec) (outputSize spec)

    -- Use default initialization for now
    let !model = MLP linear1 linear2 linear3
    return model

-- Forward pass with dropout for training and batch normalization
mlpForward :: MLP -> Tensor -> Bool -> Double -> IO Tensor
mlpForward model input isTraining dropRate = do
    -- Add input normalization for stability
    let !inputNorm = standardize input
        !x1 = Torch.tanh $ linear (linear1 model) inputNorm  -- Use tanh instead of relu
    !x1_dropped <- if isTraining then F.dropout dropRate True x1 else return x1
    let !x2 = Torch.tanh $ linear (linear2 model) x1_dropped
    !x2_dropped <- if isTraining then F.dropout dropRate True x2 else return x2
    let !output = sigmoid $ linear (linear3 model) x2_dropped
    return output
  where
    -- Simple standardization
    standardize tensor =
        let !mean = Torch.mean tensor
            !variance = Torch.mean $ mul (sub tensor mean) (sub tensor mean)
            !std = Torch.sqrt $ add variance (full' [] (1e-8 :: Float))
        in Torch.div (sub tensor mean) std

-- FIXED: Much more stable feature engineering for random embeddings
prepareEnhancedFeatures :: WordEmbedding -> WordEmbedding -> Tensor
prepareEnhancedFeatures emb1 emb2 =
    let -- Ensure embeddings are Float type
        !emb1_float = toType Float emb1
        !emb2_float = toType Float emb2

        -- More robust L2 normalization with larger epsilon
        !norm1 = Prelude.max 1e-6 (Prelude.sqrt (asValue (sumAll (mul emb1_float emb1_float)) :: Float))
        !norm2 = Prelude.max 1e-6 (Prelude.sqrt (asValue (sumAll (mul emb2_float emb2_float)) :: Float))
        !emb1_norm = Torch.div emb1_float (full' [] norm1)
        !emb2_norm = Torch.div emb2_float (full' [] norm2)

        -- Clamp normalized embeddings to prevent extreme values
        !emb1_clamped = clamp (-2.0) 2.0 emb1_norm
        !emb2_clamped = clamp (-2.0) 2.0 emb2_norm

        -- Simpler, more stable features
        !concatenated = cat (Dim 0) [emb1_clamped, emb2_clamped]
        !difference = clamp (-4.0) 4.0 $ sub emb1_clamped emb2_clamped
        !elementwise_product = clamp (-4.0) 4.0 $ mul emb1_clamped emb2_clamped

        -- More conservative similarity features
        !cosine_sim_raw = sumAll $ mul emb1_clamped emb2_clamped
        !cosine_sim = reshape [1] $ clamp (-1.0) 1.0 cosine_sim_raw

        !diff_squared = mul difference difference
        !euclidean_dist_raw = Torch.sqrt $ add (sumAll diff_squared) (full' [] (1e-8 :: Float))
        !euclidean_dist = reshape [1] $ clamp 0.0 3.0 euclidean_dist_raw

        -- Combine all features with more conservative bounds
        !combined = cat (Dim 0) [concatenated, difference, elementwise_product, cosine_sim, euclidean_dist]

        -- More robust final normalization
        !combinedSquared = mul combined combined
        !combinedNorm = Prelude.max 1e-4 (Prelude.sqrt (asValue (sumAll combinedSquared) :: Float))
        !result = clamp (-5.0) 5.0 $ Torch.div combined (full' [] combinedNorm)

        -- Debug check for the first few feature vectors (pure function)
        !finalResult = if hasNaNOrInf result
                       then Debug.Trace.trace "WARNING: NaN in feature vector!" result
                       else result

    in finalResult `seq` finalResult

-- NEW: Generate random embeddings for required words with better initialization
generateRandomEmbeddings :: Set.Set String -> Int -> IO WordEmbeddingMap
generateRandomEmbeddings requiredWords embeddingDim = do
    putStrLn $ "Generating random embeddings for " ++ show (Set.size requiredWords) ++ " words..."
    putStrLn $ "Embedding dimension: " ++ show embeddingDim

    let wordList = Set.toList requiredWords
    embeddings <- mapM generateWordEmbedding wordList
    let embeddingMap = Map.fromList (zip wordList embeddings)

    putStrLn $ "Successfully generated " ++ show (Map.size embeddingMap) ++ " random embeddings"
    putStrLn $ "Coverage: 100.0% (" ++ show (Map.size embeddingMap) ++ "/" ++ show (Set.size requiredWords) ++ " words)"

    -- Debug: Check for NaN/Inf in generated embeddings
    let sampleEmbeddings = Prelude.take 5 embeddings
    mapM_ checkEmbedding sampleEmbeddings

    performGC
    return embeddingMap
  where
    generateWordEmbedding :: String -> IO Tensor
    generateWordEmbedding word = do
        -- Use Xavier/Glorot initialization for better stability
        -- Standard deviation = sqrt(2 / (input_dim + output_dim))
        -- For embeddings, we use sqrt(2 / embedding_dim)
        let stddev = Prelude.sqrt (2.0 / fromIntegral embeddingDim)
        randomValues <- replicateM embeddingDim $ do
            -- Generate normally distributed values with proper scaling
            u1 <- randomRIO (0.0001, 0.9999) :: IO Float  -- Avoid exactly 0 or 1
            u2 <- randomRIO (0.0001, 0.9999) :: IO Float
            -- Box-Muller transform for normal distribution
            let mag = stddev * Prelude.sqrt (-2.0 * Prelude.log u1)
                normal = mag * Prelude.cos (2.0 * pi * u2)
            -- Clamp to reasonable range to prevent extreme values
            return $ Prelude.max (-2.0) $ Prelude.min 2.0 normal

        let !embedding = asTensor randomValues
        return embedding

    checkEmbedding :: Tensor -> IO ()
    checkEmbedding emb = do
        let hasNaN = hasNaNOrInf emb
            minVal = asValue (Torch.min emb) :: Float
            maxVal = asValue (Torch.max emb) :: Float
            norm = asValue (Torch.sqrt $ sumAll $ mul emb emb) :: Float
        when hasNaN $ putStrLn "WARNING: Generated embedding contains NaN/Inf!"
        when (norm < 1e-6) $ putStrLn $ "WARNING: Very small embedding norm: " ++ show norm
        when (Prelude.abs minVal > 10 || Prelude.abs maxVal > 10) $
            putStrLn $ "WARNING: Extreme embedding values: " ++ show (minVal, maxVal)

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
                            cleanW1 = trim $ map  toLower  w1
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

-- Dataset preparation with validation
prepareDataset :: WordEmbeddingMap -> [(WordPair, Label)] -> [(Tensor, Tensor)]
prepareDataset embeddings dataset =
    let results = mapMaybe convert dataset
        totalAttempted = length dataset
        successful = length results
    in Debug.Trace.trace ("Dataset conversion: " ++ show successful ++ "/" ++ show totalAttempted ++ " successful") results
  where
    convert ((w1, w2), label) = do
        e1 <- Map.lookup w1 embeddings
        e2 <- Map.lookup w2 embeddings
        let !features = prepareEnhancedFeatures e1 e2
            !target = asTensor ([if label then 1.0 else 0.0 :: Float] :: [Float])
        return (features, target)

splitDataset :: [(Tensor, Tensor)] -> Float -> ([(Tensor, Tensor)], [(Tensor, Tensor)])
splitDataset dataset ratio =
    let n = length dataset
        trainN = round $ ratio * fromIntegral n
    in splitAt trainN dataset

-- Batch utils
makeBatches :: Int -> [(Tensor, Tensor)] -> [[(Tensor, Tensor)]]
makeBatches _ [] = []
makeBatches bs xs = let (!b, !rest) = splitAt bs xs in b : makeBatches bs rest

mergeBatch :: [(Tensor, Tensor)] -> (Tensor, Tensor)
mergeBatch batch =
    let !inputs = stack (Dim 0) (map fst batch)
        !targets = reshape [-1] $ stack (Dim 0) (map snd batch)  -- Flatten to get right shape
    in (inputs, targets)

-- Training with Hasktorch Adam optimizer
trainModel :: MLP -> [(Tensor, Tensor)] -> Int -> Float -> Int -> Double -> IO MLP
trainModel model dataset epochs lr batchSize dropRate = do
    putStrLn $ "Training with " ++ show (length dataset) ++ " examples for " ++ show epochs ++ " epochs"
    putStrLn $ "Learning rate: " ++ show lr ++ ", Batch size: " ++ show batchSize ++ ", Dropout: " ++ show dropRate

    let params = flattenParameters model
        adamOptimizer = mkAdam 0 0.9 0.999 params  -- Initialize Adam with beta1=0.9, beta2=0.999

    (finalModel, _) <- foldM (trainEpoch lr batchSize dropRate dataset) (model, adamOptimizer) [1..epochs]
    return finalModel

trainEpoch :: Float -> Int -> Double -> [(Tensor, Tensor)] -> (MLP, Adam) -> Int -> IO (MLP, Adam)
trainEpoch lr batchSize dropRate dataset (currentModel, adamOptim) epoch = do
    let !batches = makeBatches batchSize dataset
    (updatedModel, updatedOptim, totalLoss, batchCount) <-
        foldM (trainBatch lr dropRate) (currentModel, adamOptim, 0.0, 0) batches

    performGC

    let !avgLoss = if batchCount > 0 then totalLoss / fromIntegral batchCount else 0.0
    printf "Epoch %d - Avg loss: %.6f (batches: %d)\n" epoch avgLoss batchCount

    -- Add more detailed debugging for first few epochs
    when (epoch <= 5) $ do
        printf "  Learning rate: %.6f, Dropout: %.2f\n" lr dropRate
        printf "  Avg loss: %.6f, Batches processed: %d\n" avgLoss batchCount
        when (isNaN avgLoss || isInfinite avgLoss || avgLoss > 10.0) $
            putStrLn "  WARNING: Loss is unstable!"

    if isNaN avgLoss || isInfinite avgLoss then do
        putStrLn "ERROR: Loss is NaN or Infinite! Training may have diverged."
        putStrLn "Try reducing learning rate or adding more gradient clipping."
        return (currentModel, adamOptim)
    else
        return (updatedModel, updatedOptim)

trainBatch :: Float -> Double -> (MLP, Adam, Float, Int) -> [(Tensor, Tensor)] -> IO (MLP, Adam, Float, Int)
trainBatch lr dropRate (model', adamOptim, accLoss, batchCount) batch = do
    let (!inputs, !targets) = mergeBatch batch

    -- Check inputs for NaN before forward pass
    if hasNaNOrInf inputs then do
        putStrLn $ "ERROR: NaN in input batch " ++ show (batchCount + 1)
        return (model', adamOptim, accLoss, batchCount + 1)
    else do
        !predictions <- mlpForward model' inputs True dropRate  -- Training mode with dropout

        -- More aggressive clamping for random embeddings
        let !clampedPredictions = clamp 1e-6 (1.0 - 1e-6) $ reshape [-1] predictions
            !clampedTargets = toType Float targets
            !lossTensor = F.binaryCrossEntropyLoss' clampedPredictions clampedTargets
            !lossScalar = asValue lossTensor :: Float

        -- More strict loss checking for random embeddings
        if isNaN lossScalar || isInfinite lossScalar || lossScalar > 10.0 then do
            putStrLn $ "WARNING: Bad loss in batch " ++ show (batchCount + 1) ++ ": " ++ show lossScalar
            putStrLn $ "Predictions range: " ++ show (asValue (Torch.min clampedPredictions) :: Float, asValue (Torch.max clampedPredictions) :: Float)
            putStrLn $ "Targets range: " ++ show (asValue (Torch.min clampedTargets) :: Float, asValue (Torch.max clampedTargets) :: Float)
            putStrLn $ "Input range: " ++ show (asValue (Torch.min inputs) :: Float, asValue (Torch.max inputs) :: Float)
            return (model', adamOptim, accLoss, batchCount + 1)
        else do
            lossScalar `seq` return ()
            predictions `seq` return ()

            let !params = flattenParameters model'
                !rawGrads = grad lossTensor params
                -- Much more aggressive gradient clipping for random embeddings
                !clippedGrads = map (clamp (-0.1) 0.1) rawGrads
                !gradients = Gradients clippedGrads

            -- Check gradients for NaN
            let gradHasNaN = Prelude.any hasNaNOrInf clippedGrads
            if gradHasNaN then do
                putStrLn $ "WARNING: NaN in gradients for batch " ++ show (batchCount + 1)
                return (model', adamOptim, accLoss, batchCount + 1)
            else do
                -- Update with Hasktorch Adam optimizer
                let !lrTensor = asTensor (lr :: Float)
                    !paramTensors = map toDependent params
                    (!updatedTensors, !updatedOptim) = adam lrTensor gradients paramTensors adamOptim
                !updatedParams <- mapM makeIndependent updatedTensors
                let !newModel = replaceParameters model' updatedParams

                -- Check updated parameters for NaN (convert IndependentTensor to Tensor)
                let !paramTensors' = map toDependent updatedParams
                    paramsHaveNaN = Prelude.any hasNaNOrInf paramTensors'
                if paramsHaveNaN then do
                    putStrLn $ "WARNING: NaN in updated parameters for batch " ++ show (batchCount + 1)
                    return (model', adamOptim, accLoss, batchCount + 1)
                else do
                    rnf newModel `seq` return ()
                    return (newModel, updatedOptim, accLoss + lossScalar, batchCount + 1)

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
    !dataset <- loadDataset "hypernym_dataset.csv"

    when (null dataset) $ do
        putStrLn "ERROR: No dataset loaded!"
        return ()

    putStrLn $ "Dataset loaded: " ++ show (length dataset) ++ " examples"

    let !requiredWords = extractRequiredWords dataset
    putStrLn $ "Found " ++ show (Set.size requiredWords) ++ " unique words"

    putStrLn "=== Generating random embeddings ==="
    let embeddingDim = 100  -- Using same dimension as original GloVe
    !embeddings <- generateRandomEmbeddings requiredWords embeddingDim

    when (Map.null embeddings) $ do
        putStrLn "ERROR: No embeddings generated!"
        return ()

    -- Additional sanity check on embeddings
    let sampleWords = Prelude.take 3 (Map.keys embeddings)
    putStrLn "=== Embedding sanity check ==="
    mapM_ (\word -> do
        case Map.lookup word embeddings of
            Just emb -> do
                let minVal = asValue (Torch.min emb) :: Float
                    maxVal = asValue (Torch.max emb) :: Float
                    norm = asValue (Torch.sqrt $ sumAll $ mul emb emb) :: Float
                printf "Word '%s': range [%.3f, %.3f], norm %.3f\n" word minVal maxVal norm
            Nothing -> return ()) sampleWords

    putStrLn "=== Preparing dataset ==="
    let !tensorData = prepareDataset embeddings dataset
    putStrLn $ "Prepared " ++ show (length tensorData) ++ " tensor examples"

    when (null tensorData) $ do
        putStrLn "ERROR: No valid dataset examples after embedding lookup."
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
        !embeddingDim' = 100  -- Random embeddings dimension
        -- FIXED: Simpler architecture for random embeddings
        !featureDim = embeddingDim' * 4 + 2
        !spec = MLPSpec featureDim 128 64 1 0.2  -- Smaller hidden layers for more stable training

    putStrLn $ "Train: " ++ show (length train) ++ ", Test: " ++ show (length test)
    putStrLn $ "Feature dimension: " ++ show featureDim

    putStrLn "=== Creating model ==="
    !model <- mkMLP spec

    putStrLn "=== Training ==="
    -- MUCH more conservative hyperparameters for random embeddings
    !trainedModel <- trainModel model train 15 0.0001 16 0.05  -- Very low LR (0.0001), smaller batch (16), minimal dropout (5%)

    putStrLn "=== Evaluation ==="
    !_ <- evaluateModel trainedModel test testPairs

    putStrLn "\n=== Interactive Predictions ==="
    putStrLn "Testing some example word pairs (note: using random embeddings):"

    predictWordPair trainedModel embeddings "animal" "dog"
    predictWordPair trainedModel embeddings "vehicle" "car"
    predictWordPair trainedModel embeddings "food" "apple"
    predictWordPair trainedModel embeddings "dog" "animal"

    when (not $ null testPairs) $ do
        putStrLn "\nTesting pairs from the test dataset:"
        let samplePairs = Prelude.take 5 testPairs
        mapM_ (\((w1, w2), _) -> predictWordPair trainedModel embeddings w1 w2) samplePairs

    putStrLn "\n=== Final cleanup ==="
    performGC
    putStrLn "Done!"
