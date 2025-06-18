{-# LANGUAGE OverloadedStrings #-}

module Main where

import WordNet.Parser
import WordNet.Distance
import WordNet.Types

import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Maybe (listToMaybe)
import Data.List (findIndex, find)

main :: IO ()
main = do
  -- Load data using functions from WordNet.Parser
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun

  putStrLn "Please enter the first noun:"
  word1Str <- getLine
  putStrLn "Please enter the second noun:"
  word2Str <- getLine

  let word1 = T.pack word1Str
      word2 = T.pack word2Str

  case calculateDistance word1 word2 idx db of
    Just dist ->
      putStrLn $ "The distance between '" ++ word1Str ++ "' and '" ++ word2Str ++ "' is: " ++ show dist
    Nothing ->
      putStrLn "Could not calculate the distance. One or both words may not be in the dictionary, or they do not share a common hypernym."

  -- Optional: Show debug information
  putStrLn "\n--- Debug Information ---"
  let debugInfo = debugWordDistance word1 word2 idx db
      debugText = formatDebugInfo word1 word2 debugInfo
  T.putStrLn debugText

-- | Interactive function to test multiple word pairs
testMultipleWords :: IO ()
testMultipleWords = do
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun

  let testPairs = [ ("cat", "dog")
                  , ("car", "automobile")
                  , ("happy", "joy")
                  , ("computer", "laptop")
                  ]

  putStrLn "Testing predefined word pairs:"
  mapM_ (testWordPair idx db) testPairs
  where
    testWordPair idx db (w1, w2) = do
      let word1 = T.pack w1
          word2 = T.pack w2
      case calculateDistance word1 word2 idx db of
        Just dist -> putStrLn $ w1 ++ " <-> " ++ w2 ++ ": " ++ show dist
        Nothing   -> putStrLn $ w1 ++ " <-> " ++ w2 ++ ": No path found"

-- | Function to get detailed distance information
getDetailedDistance :: String -> String -> IO ()
getDetailedDistance w1 w2 = do
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun

  let word1 = T.pack w1
      word2 = T.pack w2

  case calculateDistanceWithDetails word1 word2 idx db of
    Just result -> do
      putStrLn $ "Minimum distance: " ++ show (minDistance result)
      putStrLn $ "Number of synset pairs tested: " ++ show (length (synsetPairs result))
      putStrLn $ "Word 1 has " ++ show (length (distWord1Synsets result)) ++ " synsets"
      putStrLn $ "Word 2 has " ++ show (length (distWord2Synsets result)) ++ " synsets"

      putStrLn "\nAll distance calculations:"
      mapM_ printPair (synsetPairs result)
      where
        printPair (syn1, syn2, dist) =
          putStrLn $ "  " ++ show (synOffset syn1) ++ " -> " ++ show (synOffset syn2) ++ ": " ++ show dist
    Nothing ->
      putStrLn "Could not calculate detailed distance information."
