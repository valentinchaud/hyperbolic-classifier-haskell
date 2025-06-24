{-# LANGUAGE OverloadedStrings #-}

module WordNet.Hypernyms (
    getAllHypernyms,
    getHypernymWords,
    isHypernym,
    generatePositiveTriple,
    generateNegativeTriple
  ) where

import WordNet.Types
import WordNet.Parser (lookupAllSynsets)
import qualified Data.Text as T
import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.Random (randomRIO)
import Data.Maybe (mapMaybe)
import Data.List (nub)

-- | Get all hypernyms (direct and indirect) of a synset
getAllHypernyms :: SynsetDB -> Synset -> [Synset]
getAllHypernyms db synset = go Set.empty synset
  where
    go visited syn
      | synOffset syn `Set.member` visited = []  -- Cycle detection
      | otherwise =
          let newVisited = Set.insert (synOffset syn) visited
              directHypernyms = getDirectHypernyms syn
              validHypernyms = mapMaybe (`Map.lookup` db) directHypernyms
          in validHypernyms ++ concatMap (go newVisited) validHypernyms

    getDirectHypernyms syn = [offset | ('@', offset) <- synPointers syn]

-- | Get all words that are hypernyms of a given word
getHypernymWords :: T.Text -> IndexDB -> SynsetDB -> [T.Text]
getHypernymWords word idx db =
  case lookupAllSynsets word Noun idx db of
    Nothing -> []
    Just synsets ->
      let allHypernyms = concatMap (getAllHypernyms db) synsets
          hypernymWords = concatMap synWords allHypernyms
      in nub hypernymWords

-- | Check if word1 is a hypernym of word2
isHypernym :: T.Text -> T.Text -> IndexDB -> SynsetDB -> Bool
isHypernym word1 word2 idx db =
  let hypernyms = getHypernymWords word2 idx db
  in word1 `elem` hypernyms

-- | Generate a positive hypernym triple (word1, word2, 1) where word1 is hypernym of word2
generatePositiveTriple :: Vector T.Text -> IndexDB -> SynsetDB -> IO (Maybe (T.Text, T.Text, Int))
generatePositiveTriple allNouns idx db = do
  let maxIdx = V.length allNouns - 1

  -- Try up to 50 times to find a valid hypernym pair
  tryGeneratePositive maxIdx 50
  where
    tryGeneratePositive _ 0 = return Nothing
    tryGeneratePositive maxI attempts = do
      -- Pick a random word as hyponym (more specific word)
      hyponymIdx <- randomRIO (0, maxI)
      let hyponym = allNouns V.! hyponymIdx

      -- Get its hypernyms
      let hypernyms = getHypernymWords hyponym idx db

      if null hypernyms
        then tryGeneratePositive maxI (attempts - 1)
        else do
          -- Pick a random hypernym
          hypernymIdx <- randomRIO (0, length hypernyms - 1)
          let hypernym = hypernyms !! hypernymIdx

          -- Make sure the hypernym is also in our noun list
          if hypernym `V.elem` allNouns && hypernym /= hyponym
            then return $ Just (hypernym, hyponym, 1)
            else tryGeneratePositive maxI (attempts - 1)

-- | Generate a negative hypernym triple (word1, word2, 0) where word1 is NOT hypernym of word2
generateNegativeTriple :: Vector T.Text -> IndexDB -> SynsetDB -> IO (Maybe (T.Text, T.Text, Int))
generateNegativeTriple allNouns idx db = do
  let maxIdx = V.length allNouns - 1

  -- Try up to 50 times to find a valid non-hypernym pair
  tryGenerateNegative maxIdx 50
  where
    tryGenerateNegative _ 0 = return Nothing
    tryGenerateNegative maxI attempts = do
      -- Pick two random words
      idx1 <- randomRIO (0, maxI)
      idx2 <- randomRIO (0, maxI)

      let word1 = allNouns V.! idx1
          word2 = allNouns V.! idx2

      if word1 == word2
        then tryGenerateNegative maxI (attempts - 1)
        else do
          -- Check if word1 is NOT a hypernym of word2
          if not (isHypernym word1 word2 idx db)
            then return $ Just (word1, word2, 0)
            else tryGenerateNegative maxI (attempts - 1)
