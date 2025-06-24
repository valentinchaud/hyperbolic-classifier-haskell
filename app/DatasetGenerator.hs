{-# LANGUAGE OverloadedStrings #-}

module Main where

import WordNet
import WordNet.Distance
import WordNet.Hypernyms
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.IO
import System.Random (randomRIO)
import Control.Monad (replicateM_, when, forM)
import Text.Printf (printf)
import Text.Read (readMaybe)

-- | Dataset generation modes
data GenerationMode = DistanceMode | HypernymMode
  deriving (Show, Eq)

-- | Safe read function for user input
safeReadInt :: String -> Maybe Int
safeReadInt = readMaybe

-- | Get generation mode from user
getGenerationMode :: IO GenerationMode
getGenerationMode = do
  putStrLn "\nWhat type of dataset would you like to generate?"
  putStrLn "1. Distance dataset (word1, word2, distance)"
  putStrLn "2. Hypernym dataset (word1, word2, is_hypernym)"
  putStrLn "Enter your choice (1 or 2):"
  input <- getLine
  case input of
    "1" -> return DistanceMode
    "2" -> return HypernymMode
    _ -> do
      putStrLn "Invalid choice. Please enter 1 or 2."
      getGenerationMode

-- | Get number of pairs from user with error handling (for distance mode)
getNumPairs :: IO Int
getNumPairs = do
  putStrLn "\nHow many data pairs would you like to generate? (e.g., 100000)"
  input <- getLine
  case safeReadInt input of
    Just n | n > 0 -> return n
    Just _ -> do
      putStrLn "Please enter a positive number."
      getNumPairs
    Nothing -> do
      putStrLn "Invalid input. Please enter a valid number."
      getNumPairs

-- | Get number of total lines from user with error handling (for hypernym mode)
getNumLines :: IO Int
getNumLines = do
  putStrLn "\nHow many total lines would you like to generate? (Will be split 50/50 between positive and negative examples)"
  putStrLn "Enter an even number for best results (e.g., 10000):"
  input <- getLine
  case safeReadInt input of
    Just n | n > 0 -> return n
    Just _ -> do
      putStrLn "Please enter a positive number."
      getNumLines
    Nothing -> do
      putStrLn "Invalid input. Please enter a valid number."
      getNumLines

-- | Generates a single random pair, calculates distance, and writes to the file handle.
generateAndWriteDistancePair :: Handle -> Vector T.Text -> IndexDB -> SynsetDB -> IO ()
generateAndWriteDistancePair h allNouns idx db = do
  let maxIdx = V.length allNouns - 1
  -- Pick two random indices
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

-- | Generate and write a single triple to file handle
writeTriple :: Handle -> Maybe (T.Text, T.Text, Int) -> IO ()
writeTriple _ Nothing = return () -- Skip if no valid triple generated
writeTriple h (Just (word1, word2, label)) = do
  let line = T.concat [word1, ",", word2, ",", T.pack (show label)]
  T.hPutStrLn h line

-- | Generate distance dataset
generateDistanceDataset :: Int -> Vector T.Text -> IndexDB -> SynsetDB -> IO ()
generateDistanceDataset numPairs allNouns idx db = do
  let outputFile = "wordnet_distances.csv"

  withFile outputFile WriteMode $ \h -> do
    -- Write the CSV header
    T.hPutStrLn h "word1,word2,distance"
    printf "Generating %d pairs... Writing to %s\n" numPairs outputFile

    -- Loop to generate N pairs
    replicateM_ numPairs $ do
      generateAndWriteDistancePair h allNouns idx db

  printf "\nDistance dataset generation complete. Saved to %s\n" outputFile

-- | Generate balanced hypernym dataset
generateBalancedHypernymDataset :: Int -> Vector T.Text -> IndexDB -> SynsetDB -> IO ()
generateBalancedHypernymDataset totalLines allNouns idx db = do
  let numPositive = totalLines `div` 2
      numNegative = totalLines - numPositive
      outputFile = "hypernym_dataset.csv"

  withFile outputFile WriteMode $ \h -> do
    -- Write CSV header
    T.hPutStrLn h "word1,word2,is_hypernym"

    printf "Generating %d positive examples (hypernym pairs)...\n" numPositive
    positiveCount <- generateExamples h numPositive generatePositiveTriple

    printf "Generating %d negative examples (non-hypernym pairs)...\n" numNegative
    negativeCount <- generateExamples h numNegative generateNegativeTriple

    printf "\nHypernym dataset generation complete!\n"
    printf "Successfully generated %d positive and %d negative examples.\n" (positiveCount :: Int) (negativeCount :: Int)
    printf "Total: %d lines written to %s\n" (positiveCount + negativeCount) outputFile

  where
    generateExamples :: Handle -> Int -> (Vector T.Text -> IndexDB -> SynsetDB -> IO (Maybe (T.Text, T.Text, Int))) -> IO Int
    generateExamples h count generator = do
      results <- forM [1..count] $ \i -> do
        when (i `mod` 1000 == 0) $ printf "  Progress: %d/%d\n" i count
        triple <- generator allNouns idx db
        writeTriple h triple
        return $ if triple == Nothing then 0 else 1
      return $ sum results

main :: IO ()
main = do
  -- 1. Load the WordNet dictionary
  putStrLn "Loading WordNet dictionary..."
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun

  -- 2. Extract all unique nouns into an efficient Vector for random access
  let allNouns = V.fromList . map fst . filter ((== Noun) . snd) . Map.keys $ idx
  printf "Dictionary loaded with %d unique nouns.\n" (V.length allNouns)

  -- 3. Get generation mode from user
  mode <- getGenerationMode

  -- 4. Generate dataset based on selected mode
  case mode of
    DistanceMode -> do
      numPairs <- getNumPairs
      generateDistanceDataset numPairs allNouns idx db

    HypernymMode -> do
      totalLines <- getNumLines
      generateBalancedHypernymDataset totalLines allNouns idx db

  putStrLn "\nDataset generation completed successfully!"
