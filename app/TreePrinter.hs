{-# LANGUAGE OverloadedStrings #-}

module Main where

import WordNet

import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import Data.Tree (Tree(..), drawTree) -- For building and drawing the tree.
import Data.Maybe (listToMaybe)

-- | Recursively builds a hypernym tree for a given synset.
-- A hypernym is a word with a broader meaning (e.g., 'animal' is a hypernym of 'dog').
-- The WordNet data uses the '@' symbol to point to hypernyms.
buildHyperTree :: SynsetDB -> Synset -> Tree Synset
buildHyperTree db syn =
  -- A 'Node' consists of the current value (the synset) and a list of its children (sub-trees).
  Node syn
    [ buildHyperTree db nextSyn -- Recursively build the tree for each hypernym.
        | (symbol, offset) <- synPointers syn   -- For each pointer in the current synset...
        , symbol == '@'                         -- ...we only care about hypernym pointers.
        , Just nextSyn <- [Map.lookup offset db] -- Look up the synset for that pointer.
    ]

main :: IO ()
main = do
  putStrLn "Loading WordNet dictionary..."
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun
  putStrLn "Dictionary loaded."

  putStrLn "\nPlease enter a noun to see its hypernym tree (e.g., dog, car, boat):"
  wordToSearch <- T.getLine

  case Map.lookup (wordToSearch, Noun) idx of
    Nothing ->
      putStrLn $ "Sorry, the word '" ++ T.unpack wordToSearch ++ "' was not found in the noun dictionary."
    Just entry ->
      -- A word can have multiple meanings (synsets). We'll just use the first one.
      case listToMaybe (idxOffsets entry) of
        Nothing ->
          putStrLn $ "Sorry, the word '" ++ T.unpack wordToSearch ++ "' has no associated synsets."
        Just firstOffset ->
          case Map.lookup firstOffset db of
            Nothing ->
              putStrLn $ "Error: Could not find data for synset offset " ++ show firstOffset
            Just syn -> do
              putStrLn $ "\nHypernym tree for '" ++ T.unpack wordToSearch ++ "':"
              let hypernymTree = buildHyperTree db syn
              let printableTree = fmap (T.unpack . head . synWords) hypernymTree
              putStrLn $ drawTree printableTree
