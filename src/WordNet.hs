{-# LANGUAGE OverloadedStrings #-}

-- | Main WordNet module that re-exports core functionality
-- This provides the interface needed for TreeParser
module WordNet (
    -- Types and generic functions
    POS(..),
    IndexEntry(..),
    Synset(..),
    IndexDB,
    SynsetDB,
    ParseResult(..),
    -- Export record field accessors for basic types
    idxLemma,
    idxPOS,
    idxOffsets,
    synOffset,
    synPOS,
    synWords,
    synPointers,
    synGloss,
    -- Parser functions needed for TreeParser
    parseIndex,
    parseData,
    loadIndex,
    loadData,
    extractWithWarnings
  ) where

import WordNet.Types
import WordNet.Parser
