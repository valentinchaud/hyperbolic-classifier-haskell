{-# LANGUAGE OverloadedStrings #-}

module WordNet.Types (
    -- Basic types
    POS(..),
    IndexEntry(..),
    Synset(..),
    ParseResult(..),
    DistanceResult(..),
    DebugInfo(..),

    -- Type aliases
    IndexDB,
    SynsetDB,

    -- Helper functions
    isDataLine,
    safeRead,
    safeIndex,
    partitionResults
  ) where

import qualified Data.Text as T
import Data.Map (Map)

-- | Part of Speech enumeration
data POS = Noun | Verb | Adj | Adv
  deriving (Show, Eq, Ord)

-- | Index entry from WordNet index files
data IndexEntry = IndexEntry
  { idxLemma   :: T.Text
  , idxPOS     :: POS
  , idxOffsets :: [Int]
  } deriving (Show, Eq)

-- | Synset from WordNet data files
data Synset = Synset
  { synOffset   :: Int
  , synPOS      :: POS
  , synWords    :: [T.Text]
  , synPointers :: [(Char, Int)]  -- (pointer symbol, target offset)
  , synGloss    :: T.Text
  } deriving (Show, Eq)

-- | Result of parsing operations with error tracking
data ParseResult a = ParseResult
  { parsedData  :: a
  , failedLines :: [T.Text]
  , totalLines  :: Int
  } deriving (Show)

-- | Result of distance calculation with detailed information
data DistanceResult = DistanceResult
  { minDistance      :: Int
  , synsetPairs      :: [(Synset, Synset, Int)]
  , distWord1Synsets :: [Synset]
  , distWord2Synsets :: [Synset]
  } deriving (Show)

-- | Debug information for distance calculations
data DebugInfo = DebugInfo
  { word1Found         :: Bool
  , word2Found         :: Bool
  , debugWord1Synsets  :: [Synset]
  , debugWord2Synsets  :: [Synset]
  , allDistances       :: [(Synset, Synset, Maybe Int)]
  , finalDistance      :: Maybe Int
  } deriving (Show)

-- | Type aliases for database maps
type IndexDB = Map (T.Text, POS) IndexEntry
type SynsetDB = Map Int Synset

-- | Helper functions

-- | Check if a line contains data (not a comment or header)
isDataLine :: T.Text -> Bool
isDataLine line = not (T.null line) && not (T.isPrefixOf "  " line)

-- | Safe read function that returns Maybe
safeRead :: Read a => T.Text -> Maybe a
safeRead t = case reads (T.unpack t) of
  [(x, "")] -> Just x
  _         -> Nothing

-- | Safe indexing into a list
safeIndex :: [a] -> Int -> Maybe a
safeIndex xs i
  | i >= 0 && i < length xs = Just (xs !! i)
  | otherwise = Nothing

-- | Partition parsing results into successful and failed
partitionResults :: [Maybe a] -> ([a], [Int])
partitionResults xs = go xs 0 [] []
  where
    go [] _ success failures = (reverse success, reverse failures)
    go (Nothing : rest) i success failures =
      go rest (i + 1) success (i : failures)
    go (Just x : rest) i success failures =
      go rest (i + 1) (x : success) failures
