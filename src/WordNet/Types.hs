{-# LANGUAGE OverloadedStrings #-}

module WordNet.Types (
    POS(..),
    IndexEntry(..),
    Synset(..),
    IndexDB,
    SynsetDB,
    ParseResult(..),
    -- Export record field accessors explicitly
    idxLemma,
    idxPOS,
    idxOffsets,
    synOffset,
    synPOS,
    synWords,
    synPointers,
    synGloss,
    parsedData,
    failedLines,
    totalLines,
    -- Generic utility functions
    safeRead,
    safeIndex,
    isDataLine,
    partitionResults
  ) where

import qualified Data.Text as T
import Data.Map (Map)
import Text.Read (readMaybe)

-- | Part-of-speech
data POS = Noun | Verb | Adj | Adv
  deriving (Eq, Ord, Show)

-- | Index entry result
data IndexEntry = IndexEntry {
    idxLemma :: T.Text
  , idxPOS :: POS
  , idxOffsets :: [Int]
} deriving (Show, Eq)

-- | Synset from the data file
data Synset = Synset {
    synOffset :: Int
  , synPOS :: POS
  , synWords :: [T.Text]
  , synPointers :: [(Char, Int)]  -- pointer symbol + target offset
  , synGloss :: T.Text
} deriving (Show, Eq)

-- | Parse result with success/failure information
data ParseResult a = ParseResult {
    parsedData :: a
  , failedLines :: [T.Text]
  , totalLines :: Int
} deriving (Show, Eq)

type IndexDB = Map (T.Text, POS) IndexEntry
type SynsetDB = Map Int Synset

-- | Safe integer parsing
safeRead :: T.Text -> Maybe Int
safeRead = readMaybe . T.unpack

-- | Safe list indexing
safeIndex :: [a] -> Int -> Maybe a
safeIndex xs i
  | i >= 0 && i < length xs = Just (xs !! i)
  | otherwise = Nothing

-- | Filter out comment and header lines
isDataLine :: T.Text -> Bool
isDataLine line = not (T.null line) &&
                  not ("  " `T.isPrefixOf` line) &&
                  not ("#" `T.isPrefixOf` line)

-- | Helper function to partition results
partitionResults :: [Maybe a] -> ([a], [Int])
partitionResults = go 0 [] []
  where
    go _ succ fails [] = (reverse succ, reverse fails)
    go i succ fails (Just x : xs) = go (i+1) (x:succ) fails xs
    go i succ fails (Nothing : xs) = go (i+1) succ (i:fails) xs
