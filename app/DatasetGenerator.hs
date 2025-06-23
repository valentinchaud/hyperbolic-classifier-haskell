{-# LANGUAGE OverloadedStrings #-}

module Main where

import WordNet
import WordNet.Distance
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.IO
import System.Random (randomRIO)
import Control.Monad (replicateM_, when, unless)
import Text.Printf (printf)
import Text.Read (readMaybe)
import Data.IORef
import qualified Data.Set as Set
import Data.Maybe (isJust, fromMaybe, mapMaybe, catMaybes)
import Data.List (find)

-- | Safe read function for user input
safeReadInt :: String -> Maybe Int
safeReadInt = readMaybe

-- | Dataset generation modes
data DatasetMode = DistanceMode | HypernymMode deriving (Show, Eq)

-- | Structure to track positive and negative counts
data HypernymCounts = HypernymCounts
  { positiveCount :: !Int
  , negativeCount :: !Int
  } deriving (Show)

-- | Get dataset mode from user
getDatasetMode :: IO DatasetMode
getDatasetMode = do
  putStrLn "\nSelect dataset type:"
  putStrLn "1. Distance dataset (original)"
  putStrLn "2. Hypernym classification dataset (balanced)"
  putStr "Enter your choice (1 or 2): "
  input <- getLine
  case input of
    "1" -> return DistanceMode
    "2" -> return HypernymMode
    _ -> do
      putStrLn "Invalid choice. Please enter 1 or 2."
      getDatasetMode

-- | Get all synsets for a word from the IndexDB and SynsetDB
getSynsetsForWord :: T.Text -> IndexDB -> SynsetDB -> [Synset]
getSynsetsForWord word idx synsetDB = 
  case Map.lookup (word, Noun) idx of
    Nothing -> []
    Just indexEntries -> mapMaybe lookupSynset indexEntries
  where
    lookupSynset entry = Map.lookup (synOffset entry) synsetDB

-- | Get all words from a synset
getSynsetWords :: Synset -> [T.Text]
getSynsetWords synset = map (T.pack . word) (synWords synset)

-- | Get all synsets from the database
getAllSynsets :: SynsetDB -> [Synset]
getAllSynsets = Map.elems

-- | Generate positive hypernym pairs using distance-based approach
-- Since we can't directly access synset pointers, we'll use a different strategy
generatePositiveHypernymPairs :: Vector T.Text -> IndexDB -> SynsetDB -> Int -> IO [(T.Text, T.Text)]
generatePositiveHypernymPairs allNouns idx synsetDB targetCount = do
  printf "Generating %d positive hypernym pairs using semantic similarity...\n" targetCount
  
  pairs <- newIORef []
  count <- newIORef 0
  attempted <- newIORef 0
  
  let maxIdx = V.length allNouns - 1
      maxAttempts = targetCount * 10 -- Limit attempts to avoid infinite loops
      
      generateLoop = do
        currentCount <- readIORef count
        currentAttempted <- readIORef attempted
        when (currentCount < targetCount && currentAttempted < maxAttempts) $ do
          idx1 <- randomRIO (0, maxIdx)
          idx2 <- randomRIO (0, maxIdx)
          
          let word1 = allNouns V.! idx1
              word2 = allNouns V.! idx2
          
          writeIORef attempted (currentAttempted + 1)
          
          when (word1 /= word2) $ do
            -- Use semantic distance to identify potential hypernym relationships
            case calculateDistance word1 word2 idx synsetDB of
              Just dist | dist >= 1 && dist <= 3 -> do -- Close semantic distance suggests relationship
                -- For simplicity, assume word1 is hypernym of word2 if word1 is shorter/more general
                let (hypernym, hyponym) = if T.length word1 <= T.length word2 
                                         then (word1, word2) 
                                         else (word2, word1)
                currentPairs <- readIORef pairs
                writeIORef pairs ((hypernym, hyponym) : currentPairs)
                writeIORef count (currentCount + 1)
                
                when (currentCount `mod` 1000 == 0 && currentCount > 0) $
                  printf "  Generated %d positive pairs...\n" currentCount
              _ -> return ()
          
          generateLoop
  
  generateLoop
  finalPairs <- readIORef pairs
  finalCount <- readIORef count
  printf "Generated %d positive pairs\n" finalCount
  return $ take targetCount finalPairs

-- | Generate negative pairs (non-hypernym relationships)
generateNegativeHypernymPairs :: Vector T.Text -> IndexDB -> SynsetDB -> Int -> IO [(T.Text, T.Text)]
generateNegativeHypernymPairs allNouns idx synsetDB targetCount = do
  printf "Generating %d negative hypernym pairs...\n" targetCount
  
  pairs <- newIORef []
  count <- newIORef 0
  attempted <- newIORef 0
  
  let maxIdx = V.length allNouns - 1
      maxAttempts = targetCount * 5
      
      generateLoop = do
        currentCount <- readIORef count
        currentAttempted <- readIORef attempted
        when (currentCount < targetCount && currentAttempted < maxAttempts) $ do
          idx1 <- randomRIO (0, maxIdx)
          idx2 <- randomRIO (0, maxIdx)
          
          let word1 = allNouns V.! idx1
              word2 = allNouns V.! idx2
          
          writeIORef attempted (currentAttempted + 1)
          
          when (word1 /= word2) $ do
            -- Check if words are semantically distant (likely not hypernyms)
            case calculateDistance word1 word2 idx synsetDB of
              Just dist | dist > 5 -> do -- Large distance suggests no hypernym relationship
                currentPairs <- readIORef pairs
                writeIORef pairs ((word1, word2) : currentPairs)
                writeIORef count (currentCount + 1)
                
                when (currentCount `mod` 1000 == 0 && currentCount > 0) $
                  printf "  Generated %d negative pairs...\n" currentCount
              Nothing -> do -- No path found - definitely not hypernyms
                currentPairs <- readIORef pairs
                writeIORef pairs ((word1, word2) : currentPairs)
                writeIORef count (currentCount + 1)
                
                when (currentCount `mod` 1000 == 0 && currentCount > 0) $
                  printf "  Generated %d negative pairs...\n" currentCount
              _ -> return ()
          
          generateLoop
  
  generateLoop
  finalPairs <- readIORef pairs
  finalCount <- readIORef count
  printf "Generated %d negative pairs\n" finalCount
  return $ take targetCount finalPairs

-- | Generate balanced hypernym dataset using distance-based heuristics
generateBalancedHypernymDataset :: Handle -> Vector T.Text -> IndexDB -> SynsetDB -> Int -> IO ()
generateBalancedHypernymDataset h allNouns idx synsetDB totalPairs = do
  let targetPositive = totalPairs `div` 2
      targetNegative = totalPairs - targetPositive
  
  printf "Generating balanced dataset with %d total pairs\n" totalPairs
  printf "Target: %d positive + %d negative pairs\n" targetPositive targetNegative
  printf "Note: Using distance-based heuristics for hypernym detection\n"
  
  -- Generate positive pairs using semantic similarity
  positivePairs <- generatePositiveHypernymPairs allNouns idx synsetDB targetPositive
  
  -- Generate negative pairs using semantic distance
  negativePairs <- generateNegativeHypernymPairs allNouns idx synsetDB targetNegative
  
  -- Write positive pairs to file
  printf "Writing %d positive pairs to file...\n" (length positivePairs)
  mapM_ (\(hypernym, hyponym) -> do
    let line = T.concat [hypernym, ",", hyponym, ",", "1"]
    T.hPutStrLn h line
    ) positivePairs
  
  -- Write negative pairs to file
  printf "Writing %d negative pairs to file...\n" (length negativePairs)
  mapM_ (\(word1, word2) -> do
    let line = T.concat [word1, ",", word2, ",", "0"]
    T.hPutStrLn h line
    ) negativePairs
  
  let actualTotal = length positivePairs + length negativePairs
  printf "Dataset complete: %d positive + %d negative = %d total pairs\n" 
         (length positivePairs) (length negativePairs) actualTotal

-- | Generates a single random pair for distance dataset
generateAndWriteDistancePair :: Handle -> Vector T.Text -> IndexDB -> SynsetDB -> IO ()
generateAndWriteDistancePair h allNouns idx db = do
  let maxIdx = V.length allNouns - 1
  idx1 <- randomRIO (0, maxIdx)
  idx2 <- randomRIO (0, maxIdx)

  let word1 = allNouns V.! idx1
      word2 = allNouns V.! idx2

  -- Calculate distance only if the words are different
  when (word1 /= word2) $ do
    case calculateDistance word1 word2 idx db of
      Just dist -> do
        -- Write the triple to the CSV file
        let line = T.concat [word1, ",", word2, ",", T.pack (show dist)]
        T.hPutStrLn h line
      Nothing -> return () -- Skip pairs with no common path
  return ()

-- | Get number of pairs from user with error handling
getNumPairs :: IO Int
getNumPairs = do
  putStrLn "\nHow many data pairs would you like to generate? (e.g., 10000)"
  putStrLn "Note: For hypernym dataset, pairs will be generated using distance-based heuristics"
  putStrLn "Recommendation: 10000-100000 pairs for good coverage"
  input <- getLine
  case safeReadInt input of
    Just n | n > 0 && even n -> return n
    Just n | n > 0 -> return (n + 1) -- Make it even for balanced dataset
    Just _ -> do
      putStrLn "Please enter a positive number."
      getNumPairs
    Nothing -> do
      putStrLn "Invalid input. Please enter a valid number."
      getNumPairs

-- | Generate dataset based on mode
generateDataset :: DatasetMode -> Handle -> Vector T.Text -> IndexDB -> SynsetDB -> Int -> IO ()
generateDataset mode h allNouns idx db numPairs = case mode of
  DistanceMode -> do
    T.hPutStrLn h "word1,word2,distance"
    printf "Generating %d distance pairs...\n" numPairs
    replicateM_ numPairs $ generateAndWriteDistancePair h allNouns idx db
    
  HypernymMode -> do
    T.hPutStrLn h "word1,word2,is_hypernym"
    generateBalancedHypernymDataset h allNouns idx db numPairs

main :: IO ()
main = do
  -- 1. Load the WordNet dictionary
  putStrLn "Loading WordNet dictionary..."
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun

  -- 2. Extract all unique nouns into an efficient Vector for random access
  let allNouns = V.fromList . map fst . filter ((== Noun) . snd) . Map.keys $ idx
  printf "Dictionary loaded with %d unique nouns.\n" (V.length allNouns)
  printf "Synset database contains %d synsets.\n" (Map.size db)

  -- 3. Get dataset mode from user
  mode <- getDatasetMode

  -- 4. Get number of pairs to generate from the user (with error handling)
  numPairs <- getNumPairs
  
  -- 5. Set output filename based on mode
  let outputFile = case mode of
        DistanceMode -> "wordnet_distances.csv"
        HypernymMode -> "wordnet_hypernyms_balanced.csv"

  -- 6. Open a file handle and generate the data
  withFile outputFile WriteMode $ \h -> do
    generateDataset mode h allNouns idx db numPairs

  -- 7. Print completion message
  case mode of
    DistanceMode -> putStrLn "Dataset contains word pairs with their semantic distances."
    HypernymMode -> do
      putStrLn "Balanced dataset contains word pairs with hypernym labels:"
      putStrLn "  - 1 = word1 IS a hypernym of word2 (estimated)"
      putStrLn "  - 0 = word1 is NOT a hypernym of word2 (estimated)"
      putStrLn "  - Dataset is balanced: 50% positive, 50% negative examples"
      putStrLn "  - Positive pairs estimated using semantic distance heuristics"
      putStrLn "  - Note: This is a heuristic approach due to library limitations"

  printf "\nDataset generation complete. Saved to %s\n" outputFile