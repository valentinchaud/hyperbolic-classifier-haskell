{-# LANGUAGE OverloadedStrings #-}

module Test.HypernymTests (
    runAllTests,
    testCSVValidation,
    testHypernymFunctions,
    testHypernymExtraction,
    validateCSVFile,
    validateCSVFileFull,
    generateSummary,
    TestResult(..),
    testMessage,
    testPassed,
    testName
  ) where

import WordNet.Types
import WordNet.Parser (loadIndex, loadData)
import WordNet.Hypernyms (isHypernym, getHypernymWords, getAllHypernyms)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Map as Map
import Data.List (intercalate)
import Control.Monad (when, forM, forM_)
import System.IO
import Text.Printf (printf)
import Text.Read (readMaybe)
import System.Environment (lookupEnv)

-- | Test result data type
data TestResult = TestResult
  { testName :: String
  , testPassed :: Bool
  , testMessage :: String
  } deriving (Show)

-- | CSV record for validation
data CSVRecord = CSVRecord
  { csvWord1 :: T.Text
  , csvWord2 :: T.Text
  , csvLabel :: Int
  } deriving (Show, Eq)

-- | Parse a single CSV line
parseCSVLine :: T.Text -> Maybe CSVRecord
parseCSVLine line =
  case T.splitOn "," line of
    [w1, w2, labelStr] ->
      case readMaybe (T.unpack labelStr) of
        Just label -> Just $ CSVRecord w1 w2 label
        Nothing -> Nothing
    _ -> Nothing

-- | Read and parse CSV file
readCSVFile :: FilePath -> IO [CSVRecord]
readCSVFile filepath = do
  content <- T.readFile filepath
  let allLines = T.lines content
      dataLines = drop 1 allLines  -- Skip header
      parsed = map parseCSVLine dataLines
      validRecords = [r | Just r <- parsed]

  printf "Read %d valid records from %s\n" (length validRecords) filepath
  return validRecords

-- | Validate a single CSV record against WordNet
validateRecord :: IndexDB -> SynsetDB -> CSVRecord -> IO TestResult
validateRecord idx db record = do
  let word1 = csvWord1 record
      word2 = csvWord2 record
      expectedLabel = csvLabel record
      actualIsHypernym = isHypernym word1 word2 idx db
      actualLabel = if actualIsHypernym then 1 else 0

      passed = actualLabel == expectedLabel
      message = if passed
        then printf "✓ %s -> %s: Expected %d, Got %d"
                   (T.unpack word1) (T.unpack word2) expectedLabel actualLabel
        else printf "✗ %s -> %s: Expected %d, Got %d (MISMATCH)"
                   (T.unpack word1) (T.unpack word2) expectedLabel actualLabel

  return $ TestResult
    { testName = T.unpack word1 ++ " -> " ++ T.unpack word2
    , testPassed = passed
    , testMessage = message
    }

-- | Get sample size from environment or use default
getSampleSize :: Int -> IO Int
getSampleSize totalRecords = do
  maybeSampleStr <- lookupEnv "HYPERNYM_SAMPLE_SIZE"
  case maybeSampleStr of
    Just sizeStr ->
      case readMaybe sizeStr of
        Just size -> do
          printf "Using sample size from environment: %d\n" size
          return $ min size totalRecords
        Nothing -> do
          printf "Invalid HYPERNYM_SAMPLE_SIZE value: %s, using all records\n" sizeStr
          return totalRecords
    Nothing -> do
      -- Check if user wants full dataset
      maybeFullTest <- lookupEnv "HYPERNYM_FULL_TEST"
      case maybeFullTest of
        Just "true" -> do
          printf "HYPERNYM_FULL_TEST=true, testing all %d records\n" totalRecords
          return totalRecords
        _ -> do
          printf "No sample size specified, using default sample of 1000 records\n"
          printf "Set HYPERNYM_FULL_TEST=true to test all records\n"
          printf "Or set HYPERNYM_SAMPLE_SIZE=N to test N records\n"
          return $ min 1000 totalRecords

-- | Validate entire CSV file with configurable sample size
validateCSVFile :: FilePath -> IndexDB -> SynsetDB -> IO [TestResult]
validateCSVFile csvPath idx db = do
  putStrLn $ "Validating CSV file: " ++ csvPath
  records <- readCSVFile csvPath

  if null records
    then do
      putStrLn "No valid records found in CSV file"
      return []
    else do
      printf "Found %d total records\n" (length records)

      sampleSize <- getSampleSize (length records)
      let sampleRecords = take sampleSize records

      printf "Testing %d records...\n" (length sampleRecords)

      results <- forM sampleRecords $ validateRecord idx db

      return results

-- | Validate entire CSV file (all records, no sampling)
validateCSVFileFull :: FilePath -> IndexDB -> SynsetDB -> IO [TestResult]
validateCSVFileFull csvPath idx db = do
  putStrLn $ "Validating CSV file (FULL): " ++ csvPath
  records <- readCSVFile csvPath

  if null records
    then do
      putStrLn "No valid records found in CSV file"
      return []
    else do
      printf "Testing ALL %d records...\n" (length records)

      -- Add progress reporting for large datasets
      results <- forM (zip [1::Int ..] records) $ \(i, record) -> do
        when (i `mod` 1000 == 0) $
          printf "Progress: %d/%d records processed\n" i (length records)
        validateRecord idx db record

      return results

-- | Test core hypernym functions with known examples
testHypernymFunctions :: IndexDB -> SynsetDB -> IO [TestResult]
testHypernymFunctions idx db = do
  putStrLn "Testing core hypernym functions with known examples..."

  let knownPositives =
        [ ("animal", "dog")
        , ("vehicle", "car")
        , ("food", "apple")
        , ("plant", "tree")
        , ("furniture", "chair")
        ]

      knownNegatives =
        [ ("dog", "animal")  -- Reverse should be false
        , ("car", "house")   -- Unrelated
        , ("apple", "car")   -- Unrelated
        , ("chair", "dog")   -- Unrelated
        ]

  -- Test positive cases
  positiveResults <- forM knownPositives $ \(hyper, hypo) -> do
    let result = isHypernym hyper hypo idx db
        passed = result == True
        message = printf "%s is hypernym of %s: %s (%s)"
                         (T.unpack hyper) (T.unpack hypo)
                         (show result) (if passed then "✓" else "✗" :: String)

    return $ TestResult
      { testName = "Positive: " ++ T.unpack hyper ++ " -> " ++ T.unpack hypo
      , testPassed = passed
      , testMessage = message
      }

  -- Test negative cases
  negativeResults <- forM knownNegatives $ \(word1, word2) -> do
    let result = isHypernym word1 word2 idx db
        passed = result == False
        message = printf "%s is hypernym of %s: %s (%s)"
                         (T.unpack word1) (T.unpack word2)
                         (show result) (if passed then "✓" else "✗" :: String)

    return $ TestResult
      { testName = "Negative: " ++ T.unpack word1 ++ " -> " ++ T.unpack word2
      , testPassed = passed
      , testMessage = message
      }

  return $ positiveResults ++ negativeResults

-- | Test hypernym word extraction
testHypernymExtraction :: IndexDB -> SynsetDB -> IO [TestResult]
testHypernymExtraction idx db = do
  putStrLn "Testing hypernym extraction..."

  let testWords = ["dog", "car", "apple", "chair"]

  results <- forM testWords $ \word -> do
    let hypernyms = take 5 $ getHypernymWords word idx db  -- Take first 5
        hasHypernyms = not (null hypernyms)
        message = printf "Hypernyms of %s: %s"
                         (T.unpack word)
                         (intercalate ", " (map T.unpack hypernyms))

    putStrLn $ "  " ++ message

    return $ TestResult
      { testName = "Extract hypernyms: " ++ T.unpack word
      , testPassed = hasHypernyms  -- Should find at least some hypernyms
      , testMessage = message
      }

  return results

-- | Generate summary statistics for CSV validation
generateSummary :: [TestResult] -> IO ()
generateSummary results = do
  let totalTests = length results
      passedTests = length $ filter testPassed results
      failedTests = totalTests - passedTests
      successRate = if totalTests > 0
                   then (fromIntegral passedTests / fromIntegral totalTests) * 100 :: Double
                   else 0.0

  putStrLn ("\n" ++ replicate 50 '=')
  putStrLn "VALIDATION SUMMARY"
  putStrLn (replicate 50 '=')
  printf "Total tests: %d\n" totalTests
  printf "Passed: %d\n" passedTests
  printf "Failed: %d\n" failedTests
  printf "Success rate: %.2f%%\n" successRate
  putStrLn (replicate 50 '=')

  -- Show failed tests
  let failedResults = filter (not . testPassed) results
  when (not $ null failedResults) $ do
    putStrLn "\nFAILED TESTS:"
    forM_ (take 10 failedResults) $ \result -> do  -- Show first 10 failures
      putStrLn $ "  " ++ testMessage result

-- | Main test runner
runAllTests :: FilePath -> IO ()
runAllTests csvPath = do
  putStrLn "Loading WordNet dictionary for testing..."
  idx <- loadIndex "dict/index.noun" Noun
  db <- loadData "dict/data.noun" Noun

  putStrLn "Running hypernym function tests..."
  functionResults <- testHypernymFunctions idx db

  putStrLn "\nRunning hypernym extraction tests..."
  extractionResults <- testHypernymExtraction idx db

  putStrLn "\nRunning CSV validation tests..."
  csvResults <- validateCSVFile csvPath idx db

  -- Print individual test results
  putStrLn "\n=== FUNCTION TEST RESULTS ==="
  forM_ functionResults $ \result ->
    putStrLn $ testMessage result

  putStrLn "\n=== EXTRACTION TEST RESULTS ==="
  forM_ extractionResults $ \result ->
    putStrLn $ testMessage result

  -- Generate summary for CSV validation
  putStrLn "\n=== CSV VALIDATION RESULTS ==="
  generateSummary csvResults

  -- Overall summary
  let allResults = functionResults ++ extractionResults ++ csvResults
  putStrLn "\n=== OVERALL TEST SUMMARY ==="
  generateSummary allResults

-- | Quick test function for CSV validation only
testCSVValidation :: FilePath -> IO ()
testCSVValidation csvPath = do
  putStrLn $ "Quick CSV validation for: " ++ csvPath
  idx <- loadIndex "dict/index.noun" Noun
  db <- loadData "dict/data.noun" Noun

  results <- validateCSVFile csvPath idx db
  generateSummary results
