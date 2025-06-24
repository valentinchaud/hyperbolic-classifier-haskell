{-# LANGUAGE OverloadedStrings #-}

module Main where

import Test.HypernymTests
import System.Environment (getArgs)
import System.Exit (exitFailure)

main :: IO ()
main = do
  args <- getArgs
  case args of
    [csvPath] -> do
      putStrLn $ "Running tests on CSV file: " ++ csvPath
      runAllTests csvPath

    ["--quick", csvPath] -> do
      putStrLn $ "Running quick CSV validation on: " ++ csvPath
      testCSVValidation csvPath

    [] -> do
      putStrLn "Usage:"
      putStrLn "  stack exec test-runner -- path/to/hypernym_dataset.csv"
      putStrLn "  stack exec test-runner -- --quick path/to/hypernym_dataset.csv"
      putStrLn ""
      putStrLn "Examples:"
      putStrLn "  stack exec test-runner -- hypernym_dataset.csv"
      putStrLn "  stack exec test-runner -- --quick hypernym_dataset.csv"
      exitFailure

    _ -> do
      putStrLn "Invalid arguments. Use --help for usage information."
      exitFailure
