{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Tree
-- …

-- | Part-of-speech
data POS = Noun | Verb | Adj | Adv
  deriving (Eq, Ord, Show)

-- | Résultat d’index
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

            fieldsToDrop = p_cnt + 2

            offsets = drop fieldsToDrop rest
          in
            IndexEntry lemma pos (map (read . T.unpack) offsets)

      _ -> error $ "Mauvais format ou ligne trop courte: " ++ T.unpack line

-- 2. Charger toute la base d’index
loadIndex :: FilePath -> POS -> IO IndexDB
loadIndex fp pos = do
  txt <- T.readFile fp
  let rawLines = T.lines txt
      dataLines = filter (\l -> not (l `T.isPrefixOf` "  ")) rawLines
      entries  = map (parseIndexLine pos) dataLines
  return $ Map.fromList [ ((idxLemma e, idxPOS e), e) | e <- entries ]

parseDataLine :: POS -> T.Text -> Synset
parseDataLine pos line =
  let
    (beforeGloss, gloss') = T.breakOn " | " line
    gloss = T.drop 3 gloss'

    parts = T.words beforeGloss

    -- Lecture des champs de base
    synsetOffset = read (T.unpack $ head parts)
    wordCount    = read (T.unpack $ parts !! 3) :: Int

    wordFields = take (wordCount * 2) (drop 4 parts)
    words      = [wordFields !! i | i <- [0,2..length wordFields - 1]]

    pCntIndex = 4 + (wordCount * 2)
    pointerCount = read (T.unpack $ parts !! pCntIndex) :: Int

    ptrStartIndex = pCntIndex + 1
    pointers = [ (T.head (parts !! (ptrStartIndex + i*2)), read (T.unpack $ parts !! (ptrStartIndex + i*2 + 1)))
               | i <- [0..pointerCount - 1]
               ]

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
        | (sym, off) <- synPointers syn   -- bind sym here
        , sym == '@'                      -- now you can match on it
        , Just s' <- [Map.lookup off db]
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


