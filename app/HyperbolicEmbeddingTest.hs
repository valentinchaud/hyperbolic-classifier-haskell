{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import HyperbolicEmb
import Torch hiding (trace)
import qualified Torch.Functional as F
import qualified Torch.TensorFactories as TF
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.List (sortBy, minimumBy, nub)
import Data.Ord (comparing)
import Control.Monad
import Control.Exception as Control.Exception
import Data.Maybe (mapMaybe, isJust, isNothing, fromMaybe)
import Text.Printf (printf)
import Debug.Trace (trace)
import Data.Char (toLower)
import Prelude hiding (div, sqrt, max, any, mod, abs, tanh, take)
import qualified Prelude

-- ============================================================================
-- Type definitions
-- ============================================================================

type WordEmbedding = Tensor
type WordEmbeddingMap = Map.Map String Tensor

-- ============================================================================
-- GloVe loading functions (same as before but optimized for larger vocab)
-- ============================================================================

-- Optimized GloVe loader for large vocabularies
loadLargeGloVeEmbeddings :: FilePath -> Set.Set String -> IO WordEmbeddingMap
loadLargeGloVeEmbeddings filepath requiredWords = do
    putStrLn $ "Loading GloVe embeddings for " ++ show (Set.size requiredWords) ++ " words..."
    putStrLn $ "Reading from: " ++ filepath

    result <- Control.Exception.catch
        (do content <- readFile filepath
            let linesContent = lines content
            putStrLn $ "File contains " ++ show (length linesContent) ++ " lines"

            let embeddings = parseGloVeLines linesContent requiredWords Map.empty 0
            putStrLn $ "Successfully loaded " ++ show (Map.size embeddings) ++ " embeddings"

            let foundWords = Set.fromList (Map.keys embeddings)
                missingWords = Set.difference requiredWords foundWords
                coveragePercent = (fromIntegral (Set.size foundWords) * 100.0) / fromIntegral (Set.size requiredWords) :: Float

            printf "Coverage: %.1f%% (%d/%d words found)\n"
                   coveragePercent (Set.size foundWords) (Set.size requiredWords)

            when (Set.size missingWords > 0) $ do
                printf "Missing %d words\n" (Set.size missingWords)
                when (Set.size missingWords <= 20) $ do
                    putStrLn $ "Missing words: " ++ show (Prelude.take 20 $ Set.toList missingWords)

            return embeddings)
        (\e -> do
            putStrLn $ "Exception while loading GloVe file: " ++ show (e :: Control.Exception.SomeException)
            return Map.empty)
    return result

parseGloVeLines :: [String] -> Set.Set String -> WordEmbeddingMap -> Int -> WordEmbeddingMap
parseGloVeLines [] _ acc _ = acc
parseGloVeLines (line:rest) requiredWords acc processed =
    let newAcc = case words line of
                   (word:vectorStrs)
                     | Set.member word requiredWords && not (null vectorStrs) ->
                         case mapM readFloat vectorStrs of
                           Just floats ->
                             if length floats == 100
                             then Map.insert word (asTensor floats) acc
                             else acc
                           Nothing -> acc
                     | otherwise -> acc
                   _ -> acc
        newProcessed = processed + 1
    in if newProcessed `Prelude.mod` 50000 == 0
       then trace ("Processed " ++ show newProcessed ++ " lines, found " ++ show (Map.size newAcc) ++ " words") $
            parseGloVeLines rest requiredWords newAcc newProcessed
       else parseGloVeLines rest requiredWords newAcc newProcessed
  where
    readFloat :: String -> Maybe Float
    readFloat s = case reads s of
                    [(f, "")] | not (isNaN f || isInfinite f) -> Just f
                    _ -> Nothing

-- ============================================================================
-- Comprehensive test vocabulary
-- ============================================================================

-- Animals - hierarchical structure
animals :: [String]
animals = ["animal", "mammal", "vertebrate", "invertebrate",
           "dog", "cat", "horse", "cow", "pig", "sheep", "goat",
           "lion", "tiger", "elephant", "giraffe", "zebra",
           "bird", "eagle", "sparrow", "penguin", "owl", "crow",
           "fish", "shark", "salmon", "goldfish",
           "insect", "butterfly", "bee", "ant", "spider"]

-- Geography - hierarchical structure
geography :: [String]
geography = ["place", "location", "region", "area",
             "country", "city", "town", "village",
             "america", "europe", "asia", "africa",
             "france", "germany", "italy", "spain", "england",
             "paris", "london", "berlin", "rome", "madrid",
             "mountain", "river", "ocean", "lake", "forest"]

-- Objects and categories
objects :: [String]
objects = ["object", "thing", "item",
           "vehicle", "car", "truck", "bus", "bicycle", "motorcycle",
           "building", "house", "apartment", "office", "school", "hospital",
           "food", "fruit", "vegetable", "meat",
           "apple", "banana", "orange", "grape", "carrot", "potato",
           "tool", "hammer", "knife", "scissors", "computer", "phone"]

-- Abstract concepts
abstract :: [String]
abstract = ["concept", "idea", "thought", "feeling", "emotion",
            "love", "hate", "fear", "joy", "anger", "sadness",
            "good", "bad", "right", "wrong", "true", "false",
            "big", "small", "large", "tiny", "huge", "little",
            "fast", "slow", "quick", "rapid", "swift"]

-- People and roles
people :: [String]
people = ["person", "human", "people",
          "man", "woman", "child", "adult",
          "king", "queen", "prince", "princess",
          "doctor", "teacher", "student", "worker", "farmer",
          "father", "mother", "brother", "sister", "family"]

-- Actions and verbs
actions :: [String]
actions = ["action", "movement",
           "run", "walk", "jump", "climb", "swim", "fly",
           "eat", "drink", "sleep", "work", "play", "study",
           "think", "speak", "listen", "see", "hear", "feel"]

-- Create comprehensive test vocabulary
createLargeTestVocab :: [String]
createLargeTestVocab = nub $ concat [animals, geography, objects, abstract, people, actions]

-- ============================================================================
-- Advanced testing functions
-- ============================================================================

-- Convert to hyperbolic space
convertToHyperbolic :: WordEmbeddingMap -> WordEmbeddingMap
convertToHyperbolic = Map.map hyperbolicEmbeddingLayer

-- Enhanced similarity testing
findSimilarWords :: WordEmbeddingMap -> String -> Int -> [(String, Float)]
findSimilarWords embeddings queryWord k =
    case Map.lookup queryWord embeddings of
        Nothing -> []
        Just queryEmb ->
            let distances = [(word, asValue (poincareDistance queryEmb emb) :: Float)
                           | (word, emb) <- Map.toList embeddings, word /= queryWord]
                sorted = sortBy (comparing snd) distances
            in Prelude.take k sorted

-- Test hierarchical relationships in detail
testHierarchies :: WordEmbeddingMap -> IO ()
testHierarchies embeddings = do
    putStrLn "\n=== Hierarchical Structure Analysis ==="

    -- Animal hierarchy
    putStrLn "\n--- Animal Hierarchy (distance from origin) ---"
    testHierarchyGroup embeddings ["animal", "mammal", "dog", "cat", "lion", "tiger"]

    -- Geographic hierarchy
    putStrLn "\n--- Geographic Hierarchy ---"
    testHierarchyGroup embeddings ["place", "country", "city", "france", "paris"]

    -- Object hierarchy
    putStrLn "\n--- Object Hierarchy ---"
    testHierarchyGroup embeddings ["object", "vehicle", "car", "truck"]

    -- Size hierarchy
    putStrLn "\n--- Size Hierarchy ---"
    testHierarchyGroup embeddings ["big", "large", "huge", "small", "tiny"]

testHierarchyGroup :: WordEmbeddingMap -> [String] -> IO ()
testHierarchyGroup embeddings words = do
    let origin = TF.zeros [100] defaultOpts
        distances = [(word, asValue (poincareDistance origin emb) :: Float)
                   | word <- words,
                     Just emb <- [Map.lookup word embeddings]]
        sorted = sortBy (comparing snd) distances

    mapM_ (\(word, dist) -> printf "  %s: %.4f\n" word dist) sorted

-- Semantic clustering analysis
testSemanticClusters :: WordEmbeddingMap -> IO ()
testSemanticClusters embeddings = do
    putStrLn "\n=== Semantic Clustering Analysis ==="

    let testWords = ["king", "queen", "man", "woman",  -- Royalty/Gender
                     "dog", "cat", "animal", "mammal", -- Animals
                     "car", "vehicle", "truck", "bus", -- Vehicles
                     "paris", "london", "city", "france"] -- Geography

    putStrLn "\nSimilarity Matrix (Poincaré distances):"
    putStr "        "
    mapM_ (printf "%8s") testWords
    putStrLn ""

    mapM_ (\word1 -> do
        printf "%8s" word1
        mapM_ (\word2 ->
            case (Map.lookup word1 embeddings, Map.lookup word2 embeddings) of
                (Just emb1, Just emb2) -> do
                    let dist = asValue (poincareDistance emb1 emb2) :: Float
                    printf "%8.3f" dist
                _ -> printf "%8s" ("N/A" :: String)
            ) testWords
        putStrLn ""
        ) testWords

-- Validate mathematical properties
validateMathematicalProperties :: WordEmbeddingMap -> IO ()
validateMathematicalProperties embeddings = do
    putStrLn "\n=== Mathematical Properties Validation ==="

    let embList = Map.elems embeddings
        norms = map (\emb -> asValue (tensorNorm emb) :: Float) embList

    printf "Embedding Statistics:\n"
    printf "  Count: %d\n" (length norms)
    printf "  Min norm: %.6f\n" (Prelude.minimum norms)
    printf "  Max norm: %.6f\n" (Prelude.maximum norms)
    printf "  Avg norm: %.6f\n" (Prelude.sum norms / fromIntegral (length norms))
    printf "  Valid (< 1.0): %d/%d (%.1f%%)\n"
           (length $ filter (< 1.0) norms) (length norms)
           ((fromIntegral (length $ filter (< 1.0) norms) * 100.0) / fromIntegral (length norms) :: Float)

    -- Test triangle inequality (should hold for metric spaces)
    putStrLn "\n--- Testing Triangle Inequality ---"
    let testTriples = [("king", "queen", "man"), ("dog", "cat", "animal"),
                      ("car", "vehicle", "truck"), ("paris", "france", "europe")]

    mapM_ (\(a, b, c) ->
        case (Map.lookup a embeddings, Map.lookup b embeddings, Map.lookup c embeddings) of
            (Just embA, Just embB, Just embC) -> do
                let dAB = asValue (poincareDistance embA embB) :: Float
                    dBC = asValue (poincareDistance embB embC) :: Float
                    dAC = asValue (poincareDistance embA embC) :: Float
                    triangleOK = dAC <= dAB + dBC
                printf "  %s-%s--%s: %.3f <= %.3f + %.3f = %.3f %s\n"
                       a b c dAC dAB dBC (dAB + dBC) (if triangleOK then ("✅" :: String) else ("❌" :: String))
            _ -> printf "  Missing words for triangle: %s-%s-%s\n" a b c) testTriples

-- ============================================================================
-- Main comprehensive test
-- ============================================================================

main :: IO ()
main = do
    putStrLn "=== Comprehensive Large Vocabulary Hyperbolic Test ==="

    let testVocab = createLargeTestVocab
        wordSet = Set.fromList testVocab

    printf "Testing with %d words across multiple categories\n" (length testVocab)
    putStrLn $ "Categories: Animals, Geography, Objects, Abstract, People, Actions"

    -- Load embeddings
    putStrLn "\n=== Loading Large GloVe Vocabulary ==="
    glove_embeddings <- loadLargeGloVeEmbeddings "glove.6B.100d.txt" wordSet

    when (Map.null glove_embeddings) $ do
        putStrLn "ERROR: No GloVe embeddings loaded! Check file path."
        return ()

    printf "Successfully loaded %d/%d embeddings (%.1f%% coverage)\n"
           (Map.size glove_embeddings) (length testVocab)
           ((fromIntegral (Map.size glove_embeddings) * 100.0) / fromIntegral (length testVocab) :: Float)

    putStrLn "\n=== Analyzing Sample Euclidean Embeddings ==="
    mapM_ (\word ->
        case Map.lookup word glove_embeddings of
            Just emb -> analyzeEmbedding emb word
            Nothing -> putStrLn $ word ++ ": not found")
        ["king", "queen", "man", "woman", "animal", "dog"]

    -- Convert to hyperbolic space
    putStrLn "\n=== Converting to Hyperbolic Space ==="
    let hyperbolic_embeddings = convertToHyperbolic glove_embeddings

    putStrLn "\n=== Analyzing Sample Hyperbolic Embeddings ==="
    mapM_ (\word ->
        case Map.lookup word hyperbolic_embeddings of
            Just emb -> analyzeEmbedding emb word
            Nothing -> putStrLn $ word ++ ": not found")
        ["king", "queen", "man", "woman", "animal", "dog"]

    -- Comprehensive validation
    validateMathematicalProperties hyperbolic_embeddings

    -- Test hierarchical structures
    testHierarchies hyperbolic_embeddings

    -- Semantic clustering analysis
    testSemanticClusters hyperbolic_embeddings

    -- Test specific similarity cases
    putStrLn "\n=== Detailed Similarity Analysis ==="

    let queries = ["king", "animal", "paris", "car", "love", "big"]
    mapM_ (\query -> do
        putStrLn $ "\n--- Most similar to '" ++ query ++ "' ---"
        let similarities = findSimilarWords hyperbolic_embeddings query 8
        mapM_ (\(word, dist) -> printf "  %s: %.4f\n" word dist) similarities) queries

