{-# LANGUAGE OverloadedStrings #-}
module Main where
import Test.HypernymTests
import WordNet.Types
import WordNet.Parser (loadIndex, loadData)
import System.Environment (getArgs, lookupEnv)
import System.Exit (exitFailure)
import Control.Monad (when)

main :: IO ()
main = do
  -- Check if CSV path is provided via environment variable
  maybeCsvPath <- lookupEnv "HYPERNYM_CSV_PATH"

  case maybeCsvPath of
    Just csvPath -> do
      putStrLn $ "Running tests with CSV file: " ++ csvPath
      runAllTests csvPath

    Nothing -> do
      -- Try to get from command line args (for backwards compatibility)
      args <- getArgs
      case args of
        [csvPath] -> do
          putStrLn $ "Running tests on CSV file: " ++ csvPath
          runAllTests csvPath

        ["--quick", csvPath] -> do
          putStrLn $ "Running quick CSV validation on: " ++ csvPath
          testCSVValidation csvPath

        ["--full", csvPath] -> do
          putStrLn $ "Running FULL CSV validation on: " ++ csvPath
          runFullTests csvPath

        [] -> do
          putStrLn "No CSV file specified."
          putStrLn "Usage options:"
          putStrLn "  1. Environment variables:"
          putStrLn "     HYPERNYM_CSV_PATH=dataset.csv stack test"
          putStrLn "     HYPERNYM_FULL_TEST=true HYPERNYM_CSV_PATH=dataset.csv stack test"
          putStrLn "     HYPERNYM_SAMPLE_SIZE=10000 HYPERNYM_CSV_PATH=dataset.csv stack test"
          putStrLn ""
          putStrLn "  2. Command line arguments:"
          putStrLn "     stack test --test-arguments='dataset.csv'"
          putStrLn "     stack test --test-arguments='--quick dataset.csv'"
          putStrLn "     stack test --test-arguments='--full dataset.csv'"
          putStrLn ""

          -- Run basic tests without CSV
          putStrLn "Running basic hypernym function tests without CSV validation..."
          runBasicTests

        _ -> do
          putStrLn "Invalid arguments."
          putStrLn "Valid options: [csvfile], [--quick csvfile], [--full csvfile]"
          exitFailure

-- | Run tests with full CSV validation (all records)
runFullTests :: FilePath -> IO ()
runFullTests csvPath = do
  putStrLn "Loading WordNet dictionary for testing..."
  idx <- loadIndex "dict/index.noun" Noun
  db <- loadData "dict/data.noun" Noun

  putStrLn "Running hypernym function tests..."
  functionResults <- testHypernymFunctions idx db

  putStrLn "\nRunning hypernym extraction tests..."
  extractionResults <- testHypernymExtraction idx db

  putStrLn "\nRunning FULL CSV validation tests..."
  csvResults <- validateCSVFileFull csvPath idx db

  -- Print individual test results
  putStrLn "\n=== FUNCTION TEST RESULTS ==="
  mapM_ (putStrLn . testMessage) functionResults

  putStrLn "\n=== EXTRACTION TEST RESULTS ==="
  mapM_ (putStrLn . testMessage) extractionResults

  -- Generate summary for CSV validation
  putStrLn "\n=== FULL CSV VALIDATION RESULTS ==="
  generateSummary csvResults

  -- Overall summary
  let allResults = functionResults ++ extractionResults ++ csvResults
  putStrLn "\n=== OVERALL TEST SUMMARY ==="
  generateSummary allResults

-- | Run basic tests without CSV file
runBasicTests :: IO ()
runBasicTests = do
  putStrLn "Loading WordNet dictionary for basic testing..."
  idx <- loadIndex "dict/index.noun" Noun
  db <- loadData "dict/data.noun" Noun

  putStrLn "Running hypernym function tests..."
  functionResults <- testHypernymFunctions idx db

  putStrLn "\nRunning hypernym extraction tests..."
  extractionResults <- testHypernymExtraction idx db

  -- Print results
  putStrLn "\n=== FUNCTION TEST RESULTS ==="
  mapM_ (putStrLn . testMessage) functionResults

  putStrLn "\n=== EXTRACTION TEST RESULTS ==="
  mapM_ (putStrLn . testMessage) extractionResults

  -- Summary
  let allResults = functionResults ++ extractionResults
  putStrLn "\n=== TEST SUMMARY ==="
  generateSummary allResults
