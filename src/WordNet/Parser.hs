{-# LANGUAGE OverloadedStrings #-}

module WordNet.Parser (
    parseIndex,
    parseData,
    getPathToRoot,
    formatSynset,
    formatDebugInfo,
    loadIndex,
    loadData,
    extractWithWarnings,
    lookupAllSynsets
  ) where

import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Map (Map)
import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Maybe (mapMaybe)
import Control.Exception (try, SomeException)
import Control.Monad (when)

import WordNet.Types

-- | Simple but robust index line parser
parseIndexLine :: POS -> T.Text -> Maybe IndexEntry
parseIndexLine pos line =
  case T.words line of
    (lemma : _ : _ : p_cnt_str : rest) ->
      case safeRead p_cnt_str of
        Just p_cnt ->
          let fieldsToDrop = p_cnt + 2
              offsetStrings = drop fieldsToDrop rest
              offsets = mapMaybe safeRead offsetStrings
          in if not (null offsets)
             then Just $ IndexEntry lemma pos offsets
             else Nothing
        Nothing -> Nothing
    _ -> Nothing

parseDataLine :: POS -> T.Text -> Maybe Synset
parseDataLine pos line = do
  let (beforeGloss, afterSep) = T.breakOn " | " line
      gloss = if T.null afterSep then "" else T.drop 3 afterSep
      parts = T.words beforeGloss

  -- Parse synset offset (first field)
  offsetStr <- safeIndex parts 0
  synsetOffset <- safeRead offsetStr

  -- Parse word count (4th field, 0-indexed as 3)
  wordCountStr <- safeIndex parts 3
  wordCount <- safeRead wordCountStr

  -- Extract words
  let wordFields = take (wordCount * 2) (drop 4 parts)
      words = [wordFields !! i | i <- [0,2..length wordFields - 1]]

  -- Parse pointer count
  let pCntIndex = 4 + (wordCount * 2)
  pointerCountStr <- safeIndex parts pCntIndex
  pointerCount <- safeRead pointerCountStr

  -- Parse pointers
  let ptrStartIndex = pCntIndex + 1
      parsePointer i = do
        ptrSymbol <- safeIndex parts (ptrStartIndex + i*2)
        ptrOffsetStr <- safeIndex parts (ptrStartIndex + i*2 + 1)
        ptrOffset <- safeRead ptrOffsetStr
        return (T.head ptrSymbol, ptrOffset)

      pointers = mapMaybe parsePointer [0..pointerCount - 1]

  -- Only succeed if we parsed all expected pointers
  if length pointers == pointerCount
    then Just $ Synset synsetOffset pos words pointers gloss
    else Nothing

parseIndex :: POS -> T.Text -> ParseResult IndexDB
parseIndex pos content =
  let rawLines = T.lines content
      dataLines = filter isDataLine rawLines
      (successfulEntries, failedLines) = partitionResults $ map (parseIndexLine pos) dataLines
      indexDB = Map.fromList [ ((idxLemma e, idxPOS e), e) | e <- successfulEntries ]
  in ParseResult indexDB (map (dataLines !!) failedLines) (length dataLines)

parseData :: POS -> T.Text -> ParseResult SynsetDB
parseData pos content =
  let linesData = filter (not . T.null) (T.lines content)
      (successfulSynsets, failedLines) = partitionResults $ map (parseDataLine pos) linesData
      synsetDB = Map.fromList [(synOffset s, s) | s <- successfulSynsets]
  in ParseResult synsetDB (map (linesData !!) failedLines) (length linesData)

getPathToRoot :: SynsetDB -> Synset -> [Synset]
getPathToRoot db = go Set.empty
  where
    go visited syn
      | synOffset syn `Set.member` visited = [syn]  -- Cycle detection
      | otherwise =
          let newVisited = Set.insert (synOffset syn) visited
          in case getHypernymSynset syn of
               Just nextSyn -> syn : go newVisited nextSyn
               Nothing      -> [syn]

    getHypernymSynset syn = do
      hypernymOffset <- lookup '@' (synPointers syn)
      Map.lookup hypernymOffset db

lookupAllSynsets :: T.Text -> POS -> IndexDB -> SynsetDB -> Maybe [Synset]
lookupAllSynsets word pos idx db = do
  entry <- Map.lookup (word, pos) idx
  let offsets = idxOffsets entry
      synsets = [s | offset <- offsets, Just s <- [Map.lookup offset db]]
  if null synsets then Nothing else Just synsets

formatSynset :: Synset -> T.Text
formatSynset s = T.pack (show (synOffset s)) <> ": " <>
                 T.intercalate ", " (synWords s) <> " | " <> synGloss s

formatDebugInfo :: T.Text -> T.Text -> DebugInfo -> T.Text
formatDebugInfo word1 word2 debug =
  let header = "=== Debug for: " <> word1 <> " vs " <> word2 <> " ==="

      word1Section = if word1Found debug
        then word1 <> " synsets:\n" <>
             T.unlines ["  " <> formatSynset s | s <- debugWord1Synsets debug]
        else word1 <> " not found in index\n"

      word2Section = if word2Found debug
        then word2 <> " synsets:\n" <>
             T.unlines ["  " <> formatSynset s | s <- debugWord2Synsets debug]
        else word2 <> " not found in index\n"

      distanceSection = if word1Found debug && word2Found debug
        then "\nDistances between all synset pairs:\n" <>
             T.unlines [ "  " <> T.pack (show (synOffset s1)) <> " -> " <>
                        T.pack (show (synOffset s2)) <> ": " <>
                        maybe "No path" (T.pack . show) dist
                       | (s1, s2, dist) <- allDistances debug
                       ] <>
             "\nMinimum distance: " <> maybe "No path" (T.pack . show) (finalDistance debug) <> "\n"
        else ""

  in T.unlines [header, word1Section, word2Section, distanceSection]

-- | Helper functions for working with ParseResult in IO context

-- | Extract data from ParseResult, printing warnings about failed parses
extractWithWarnings :: String -> ParseResult a -> IO a
extractWithWarnings filename result = do
  let failedCount = length (failedLines result)
  when (failedCount > 0) $
    putStrLn $ "Warning: Failed to parse " ++ show failedCount ++
               " lines in " ++ filename
  return (parsedData result)

-- | Load and parse index file (IO wrapper around pure functions)
loadIndex :: FilePath -> POS -> IO IndexDB
loadIndex fp pos = do
  result <- try (T.readFile fp) :: IO (Either SomeException T.Text)
  case result of
    Left ex -> do
      putStrLn $ "Error reading file " ++ fp ++ ": " ++ show ex
      return Map.empty
    Right content -> do
      parseResult <- return $ parseIndex pos content
      extractWithWarnings fp parseResult

-- | Load and parse data file (IO wrapper around pure functions)
loadData :: FilePath -> POS -> IO SynsetDB
loadData fp pos = do
  result <- try (T.readFile fp) :: IO (Either SomeException T.Text)
  case result of
    Left ex -> do
      putStrLn $ "Error reading file " ++ fp ++ ": " ++ show ex
      return Map.empty
    Right content -> do
      parseResult <- return $ parseData pos content
      extractWithWarnings fp parseResult
