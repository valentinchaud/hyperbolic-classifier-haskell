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
    lookupSynset entry = Map.lookup (synsetOffset entry) synsetDB

-- | Get hypernym offsets from a synset's pointer list
getHypernymOffsets :: Synset -> [SynsetOffset]
getHypernymOffsets synset = 
  [target | Pointer Hypernym target _ _ <- synsetPointers synset]

-- | Get hyponym offsets from a synset's pointer list
getHyponymOffsets :: Synset -> [SynsetOffset]
getHyponymOffsets synset = 
  [target | Pointer Hyponym target _ _ <- synsetPointers synset]

-- | Get all words from a synset
getSynsetWords :: Synset -> [T.Text]
getSynsetWords synset = map wordForm (synsetWords synset)

-- | Get all synsets from the database
getAllSynsets :: SynsetDB -> [Synset]
getAllSynsets = Map.elems

-- | Generate positive hypernym pairs directly from synset relationships
generatePositiveHypernymPairs :: SynsetDB -> Int -> IO [(T.Text, T.Text)]
generatePositiveHypernymPairs synsetDB targetCount = do
  let allSynsets = getAllSynsets synsetDB
  printf "Generating %d positive hypernym pairs from %d synsets...\n" targetCount (length allSynsets)
  
  pairs <- newIORef []
  count <- newIORef 0
  
  -- Process each synset to find hypernym relationships
  mapM_ (processSynsetForHypernyms synsetDB pairs count targetCount) allSynsets
  
  finalPairs <- readIORef pairs
  finalCount <- readIORef count
  printf "Generated %d positive pairs\n" finalCount
  return $ take targetCount finalPairs

-- | Process a synset to extract hypernym pairs
processSynsetForHypernyms :: SynsetDB -> IORef [(T.Text, T.Text)] -> IORef Int -> Int -> Synset -> IO ()
processSynsetForHypernyms synsetDB pairsRef countRef targetCount synset = do
  currentCount <- readIORef countRef
  when (currentCount < targetCount) $ do
    let hypernymOffsets = getHypernymOffsets synset
        hyponymWords = getSynsetWords synset
    
    -- For each hypernym, create pairs with hyponym words
    mapM_ (createHypernymPairs synsetDB pairsRef countRef targetCount hyponymWords) hypernymOffsets

-- | Create hypernym pairs for a specific hypernym offset
createHypernymPairs :: SynsetDB -> IORef [(T.Text, T.Text)] -> IORef Int -> Int -> [T.Text] -> SynsetOffset -> IO ()
createHypernymPairs synsetDB pairsRef countRef targetCount hyponymWords hypernymOffset = do
  currentCount <- readIORef countRef
  when (currentCount < targetCount) $ do
    case Map.lookup hypernymOffset synsetDB of
      Nothing -> return ()
      Just hypernymSynset -> do
        let hypernymWords = getSynsetWords hypernymSynset
        -- Create pairs: hypernym -> hyponym
        mapM_ (addHypernymPair pairsRef countRef targetCount) 
              [(hw, ho) | hw <- hypernymWords, ho <- hyponymWords, hw /= ho]

-- | Add a hypernym pair to the collection
addHypernymPair :: IORef [(T.Text, T.Text)] -> IORef Int -> Int -> (T.Text, T.Text) -> IO ()
addHypernymPair pairsRef countRef targetCount pair = do
  currentCount <- readIORef countRef
  when (currentCount < targetCount) $ do
    currentPairs <- readIORef pairsRef
    writeIORef pairsRef (pair : currentPairs)
    writeIORef countRef (currentCount + 1)
    
    when (currentCount `mod` 1000 == 0 && currentCount > 0) $
      printf "  Generated %d positive pairs...\n" currentCount

-- | Generate negative pairs (non-hypernym relationships)
generateNegativeHypernymPairs :: Vector T.Text -> IndexDB -> SynsetDB -> Int -> IO [(T.Text, T.Text)]
generateNegativeHypernymPairs allNouns idx synsetDB targetCount = do
  printf "Generating %d negative hypernym pairs...\n" targetCount
  
  pairs <- newIORef []
  count <- newIORef 0
  
  let maxIdx = V.length allNouns - 1
      generateLoop = do
        currentCount <- readIORef count
        when (currentCount < targetCount) $ do
          idx1 <- randomRIO (0, maxIdx)
          idx2 <- randomRIO (0, maxIdx)
          
          let word1 = allNouns V.! idx1
              word2 = allNouns V.! idx2
          
          when (word1 /= word2) $ do
            -- Check if it's NOT a hypernym relationship
            if not (isDirectHypernym word1 word2 idx synsetDB)
              then do
                currentPairs <- readIORef pairs
                writeIORef pairs ((word1, word2) : currentPairs)
                writeIORef count (currentCount + 1)
                
                when (currentCount `mod` 1000 == 0 && currentCount > 0) $
                  printf "  Generated %d negative pairs...\n" currentCount
              else return ()
          
          generateLoop
  
  generateLoop
  finalPairs <- readIORef pairs
  finalCount <- readIORef count
  printf "Generated %d negative pairs\n" finalCount
  return $ take targetCount finalPairs

-- | Check if word1 is a direct hypernym of word2 (only immediate parent)
isDirectHypernym :: T.Text -> T.Text -> IndexDB -> SynsetDB -> Bool
isDirectHypernym word1 word2 idx synsetDB = 
  let synsets1 = getSynsetsForWord word1 idx synsetDB
      synsets2 = getSynsetsForWord word2 idx synsetDB
      offsets1 = Set.fromList $ map synsetOffset synsets1
  in any (\synset2 -> 
           let directHypernymOffsets = Set.fromList $ getHypernymOffsets synset2
           in not $ Set.null $ Set.intersection offsets1 directHypernymOffsets
         ) synsets2

-- | Generate balanced hypernym dataset using synset relationships
generateBalancedHypernymDataset :: Handle -> Vector T.Text -> IndexDB -> SynsetDB -> Int -> IO ()
generateBalancedHypernymDataset h allNouns idx synsetDB totalPairs = do
  let targetPositive = totalPairs `div` 2
      targetNegative = totalPairs - targetPositive
  
  printf "Generating balanced dataset with %d total pairs\n" totalPairs
  printf "Target: %d positive + %d negative pairs\n" targetPositive targetNegative
  
  -- Generate positive pairs using synset relationships
  positivePairs <- generatePositiveHypernymPairs synsetDB targetPositive
  
  -- Generate negative pairs randomly
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
  putStrLn "Note: For hypernym dataset, pairs will be generated directly from WordNet synset relationships"
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
      putStrLn "  - 1 = word1 IS a hypernym of word2"
      putStrLn "  - 0 = word1 is NOT a hypernym of word2"
      putStrLn "  - Dataset is balanced: 50% positive, 50% negative examples"
      putStrLn "  - Positive pairs generated directly from WordNet synset relationships"

  printf "\nDataset generation complete. Saved to %s\n" outputFile