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
import Control.Monad (replicateM_, when)
import Text.Printf (printf)
import Text.Read (readMaybe)

-- | Safe read function for user input
safeReadInt :: String -> Maybe Int
safeReadInt = readMaybe

-- | Generates a single random pair, calculates distance, and writes to the file handle.
generateAndWritePair :: Handle -> Vector T.Text -> IndexDB -> SynsetDB -> IO ()
generateAndWritePair h allNouns idx db = do
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

-- | Get number of pairs from user with error handling
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

main :: IO ()
main = do
  -- 1. Load the WordNet dictionary
  putStrLn "Loading WordNet dictionary..."
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun

  -- 2. Extract all unique nouns into an efficient Vector for random access
  let allNouns = V.fromList . map fst . filter ((== Noun) . snd) . Map.keys $ idx
  printf "Dictionary loaded with %d unique nouns.\n" (V.length allNouns)

  -- 3. Get number of pairs to generate from the user (with error handling)
  numPairs <- getNumPairs
  let outputFile = "wordnet_distances.csv"

  -- 4. Open a file handle and generate the data
  withFile outputFile WriteMode $ \h -> do
    -- Write the CSV header
    T.hPutStrLn h "word1,word2,distance"
    printf "Generating %d pairs... Writing to %s\n" numPairs outputFile

    -- Loop to generate N pairs
    replicateM_ numPairs $ do
      generateAndWritePair h allNouns idx db

  printf "\nDataset generation complete. Saved to %s\n" outputFile
