{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Tree

-- | Part-of-speech
data POS = Noun | Verb | Adj | Adv
  deriving (Eq, Ord, Show)

-- | Index entry result
data IndexEntry = IndexEntry {
    idxLemma :: T.Text
  , idxPOS :: POS
  , idxOffsets :: [Int]
} deriving Show

data Synset = Synset {
    synOffset :: Int
  , synPOS :: POS
  , synWords :: [T.Text]
  , synPointers :: [(Char, Int)]  -- pointer symbol + target offset
  , synGloss :: T.Text
} deriving Show

type IndexDB = Map (T.Text, POS) IndexEntry
type SynsetDB = Map Int Synset

parseIndexLine :: POS -> T.Text -> IndexEntry
parseIndexLine pos line =
  let parts = T.words line
  in case parts of
      (lemma : _ : _ : p_cnt_str : rest) ->
          let
            p_cnt = read (T.unpack p_cnt_str) :: Int
            fieldsToDrop = p_cnt + 2  -- skip pointer fields and two more fields
            offsets = drop fieldsToDrop rest  -- get offsets after pointer fields
          in
            IndexEntry lemma pos (map (read . T.unpack) offsets)
      _ -> error $ "Bad format or line too short: " ++ T.unpack line

loadIndex :: FilePath -> POS -> IO IndexDB
loadIndex fp pos = do
  txt <- T.readFile fp
  let rawLines = T.lines txt
      dataLines = filter (\l -> not (l `T.isPrefixOf` "  ")) rawLines  -- skip comment lines
      entries  = map (parseIndexLine pos) dataLines
  return $ Map.fromList [ ((idxLemma e, idxPOS e), e) | e <- entries ]

parseDataLine :: POS -> T.Text -> Synset
parseDataLine pos line =
  let
    (beforeGloss, gloss') = T.breakOn " | " line
    gloss = T.drop 3 gloss'  -- remove " | " prefix

    parts = T.words beforeGloss

    synsetOffset = read (T.unpack $ head parts)
    wordCount    = read (T.unpack $ parts !! 3) :: Int

    wordFields = take (wordCount * 2) (drop 4 parts)
    words      = [wordFields !! i | i <- [0,2..length wordFields - 1]]  -- take every other field (the words)

    pCntIndex = 4 + (wordCount * 2)
    pointerCount = read (T.unpack $ parts !! pCntIndex) :: Int

    ptrStartIndex = pCntIndex + 1
    pointers = [ (T.head (parts !! (ptrStartIndex + i*2)), read (T.unpack $ parts !! (ptrStartIndex + i*2 + 1)))
               | i <- [0..pointerCount - 1]
               ]  -- parse pointer symbol and offset pairs

  in Synset synsetOffset pos words pointers gloss

loadData :: FilePath -> POS -> IO SynsetDB
loadData fp pos = do
  linesData <- T.lines <$> T.readFile fp
  let syns = map (parseDataLine pos) (filter (not . T.null) linesData)
  return $ Map.fromList [(synOffset s, s) | s <- syns]

buildHyperTree :: SynsetDB -> Synset -> Tree Synset
buildHyperTree db syn =
  Node syn
    [ buildHyperTree db s'
        | (sym, off) <- synPointers syn   -- for each pointer
        , sym == '@'                      -- only hypernym pointers
        , Just s' <- [Map.lookup off db]  -- lookup target synset
    ]

main :: IO ()
main = do
  idx <- loadIndex "dict/index.noun" Noun
  db  <- loadData  "dict/data.noun" Noun
  putStrLn "Please enter a word to search"
  wordToSearch <- getLine
  let keyToSearch = T.pack wordToSearch
      Just entry = Map.lookup (keyToSearch, Noun) idx
  case idxOffsets entry of
       (off:_) -> do
         putStrLn $ "Processing offset: " ++ show off
         case Map.lookup off db of
           Just syn -> do
             let tree = buildHyperTree db syn
             putStrLn . drawTree $ fmap (T.unpack . head . synWords) tree
           Nothing ->
             putStrLn $ "Error: Offset " ++ show off ++ " not found in data.noun."
       [] -> do
         putStrLn $ "Warning: The word '" ++ T.unpack (idxLemma entry) ++ "' has no offsets in the database. Cannot build tree."
