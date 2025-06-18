{-# LANGUAGE OverloadedStrings #-}

module WordNet.Distance (
    calculateDistance,
    calculateDistanceWithDetails,
    debugWordDistance,
    calculateSynsetDistance,
    findLCA,
    distanceToLCA
  ) where

import qualified Data.Text as T
import qualified Data.Set as Set
import Data.List (findIndex, find)
import Data.Maybe (mapMaybe)

import WordNet.Types
import WordNet.Parser (getPathToRoot, lookupAllSynsets)

-- | Find the Lowest Common Ancestor (LCA) between two paths
findLCA :: [Synset] -> [Synset] -> Maybe Synset
findLCA path1 path2 =
  let path2Offsets = Set.fromList (map synOffset path2)
  in find (\s -> synOffset s `Set.member` path2Offsets) path1

-- | Calculate the number of steps from a start node to the LCA node in a given path
distanceToLCA :: Synset -> [Synset] -> Maybe Int
distanceToLCA lca path = findIndex (== lca) path

-- | Calculate distance between two specific synsets
calculateSynsetDistance :: SynsetDB -> Synset -> Synset -> Maybe Int
calculateSynsetDistance db syn1 syn2 = do
  let path1 = getPathToRoot db syn1
      path2 = getPathToRoot db syn2
  lca <- findLCA path1 path2
  d1 <- distanceToLCA lca path1
  d2 <- distanceToLCA lca path2
  return (d1 + d2)

-- | Calculate the minimum semantic distance between two words (purely functional)
calculateDistance :: T.Text -> T.Text -> IndexDB -> SynsetDB -> Maybe Int
calculateDistance word1 word2 idx db = do
  synsets1 <- lookupAllSynsets word1 Noun idx db
  synsets2 <- lookupAllSynsets word2 Noun idx db

  let distances = [ calculateSynsetDistance db syn1 syn2
                  | syn1 <- synsets1
                  , syn2 <- synsets2
                  ]
      validDistances = [d | Just d <- distances]

  case validDistances of
    [] -> Nothing
    ds -> Just (minimum ds)

-- | Calculate distance with detailed information (purely functional)
calculateDistanceWithDetails :: T.Text -> T.Text -> IndexDB -> SynsetDB -> Maybe DistanceResult
calculateDistanceWithDetails word1 word2 idx db = do
  synsets1 <- lookupAllSynsets word1 Noun idx db
  synsets2 <- lookupAllSynsets word2 Noun idx db

  let validPairs = [(s1, s2, d) | s1 <- synsets1, s2 <- synsets2, Just d <- [calculateSynsetDistance db s1 s2]]

  case validPairs of
    [] -> Nothing
    pairs ->
      let minDist = minimum [d | (_, _, d) <- pairs]
      in Just $ DistanceResult minDist pairs synsets1 synsets2

-- | Generate debug information for word distance calculation (purely functional)
debugWordDistance :: T.Text -> T.Text -> IndexDB -> SynsetDB -> DebugInfo
debugWordDistance word1 word2 idx db =
  let mSynsets1 = lookupAllSynsets word1 Noun idx db
      mSynsets2 = lookupAllSynsets word2 Noun idx db

      synsets1 = maybe [] id mSynsets1
      synsets2 = maybe [] id mSynsets2

      allDistances = [ (syn1, syn2, calculateSynsetDistance db syn1 syn2)
                     | syn1 <- synsets1
                     , syn2 <- synsets2
                     ]

      finalDistance = calculateDistance word1 word2 idx db

  in DebugInfo {
       word1Found = maybe False (const True) mSynsets1
     , word2Found = maybe False (const True) mSynsets2
     , debugWord1Synsets = synsets1
     , debugWord2Synsets = synsets2
     , allDistances = allDistances
     , finalDistance = finalDistance
     }
