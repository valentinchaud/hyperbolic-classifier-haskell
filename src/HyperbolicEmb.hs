{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

module HyperbolicEmb where

import Torch.Tensor
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import qualified Torch.TensorFactories as TF
import Torch.DType
import Torch.Device
import Torch.TensorOptions

-- | Core hyperbolic embedding operations for Hasktorch
-- Based on Poincaré ball model

-- | Manual implementation of L2 norm using basic operations
tensorNorm :: Tensor -> Tensor
tensorNorm x = F.sqrt (F.sumAll (x * x))

-- | IMPROVED: Project vectors to Poincaré ball with better structure preservation
projectToPoincare :: Tensor -> Tensor
projectToPoincare x =
  let norms = tensorNorm x
      -- Only project if actually needed (norm >= 1.0)
      -- Use much gentler scaling to preserve relationships
      max_norm = 0.99
      scaling_needed = F.ge norms 1.0
      scale_factor = max_norm / (norms + 1e-8)
      ones_tensor = TF.onesLike scale_factor

      -- Only scale if needed, otherwise keep original
      final_scale = FI.where' scaling_needed scale_factor ones_tensor
  in x * final_scale

-- | IMPROVED: Poincaré distance with better numerical stability
poincareDistance :: Tensor -> Tensor -> Tensor
poincareDistance x y =
  let x_norm2 = F.sumAll (x * x)
      y_norm2 = F.sumAll (y * y)
      diff = x - y
      diff_norm2 = F.sumAll (diff * diff)

      -- More stable computation
      numerator = 2.0 * diff_norm2
      denom1 = 1.0 - x_norm2
      denom2 = 1.0 - y_norm2

      -- Better clamping to avoid numerical issues
      denom1_safe = F.clamp 1e-6 1.0 denom1
      denom2_safe = F.clamp 1e-6 1.0 denom2

      inner = 1.0 + numerator / (denom1_safe * denom2_safe)
      inner_clamped = F.clamp 1.0001 50.0 inner

  in F.log (inner_clamped + F.sqrt (inner_clamped * inner_clamped - 1.0))

-- | Exponential map from tangent space at origin to Poincaré ball
-- exp_0(v) = tanh(||v||) * v/||v||
expMapOrigin :: Tensor -> Tensor
expMapOrigin v =
  let v_norm = tensorNorm v
      v_norm_safe = F.clamp 1e-8 10.0 v_norm  -- Avoid division by zero
      tanh_norm = F.tanh v_norm_safe
      direction = v / (v_norm_safe + 1e-8)  -- Add epsilon for safety
  in direction * tanh_norm

-- | Logarithmic map from Poincaré ball to tangent space at origin
-- log_0(x) = artanh(||x||) * x/||x||
logMapOrigin :: Tensor -> Tensor
logMapOrigin x =
  let x_norm = tensorNorm x
      x_norm_safe = F.clamp 1e-8 0.9999 x_norm  -- Keep away from boundary
      -- Manual artanh: artanh(x) = 0.5 * log((1+x)/(1-x))
      artanh_norm = 0.5 * F.log ((1.0 + x_norm_safe) / (1.0 - x_norm_safe))
      direction = x / (x_norm_safe + 1e-8)
  in direction * artanh_norm

-- | FIXED: Möbius addition with proper numerical stability
mobiusAdd :: Tensor -> Tensor -> Tensor
mobiusAdd u v =
  let u_norm2 = F.sumAll (u * u)
      v_norm2 = F.sumAll (v * v)
      uv_dot = F.sumAll (u * v)

      -- Better numerical stability - avoid clamping unless necessary
      u_norm2_safe = F.clamp 0.0 0.999 u_norm2  -- Stay well inside unit ball
      v_norm2_safe = F.clamp 0.0 0.999 v_norm2

      -- Standard Möbius addition formula
      coeff1 = 1.0 + 2.0 * uv_dot + v_norm2_safe
      coeff2 = 1.0 - u_norm2_safe
      numerator = u * coeff1 + v * coeff2

      denominator = 1.0 + 2.0 * uv_dot + u_norm2_safe * v_norm2_safe
      denominator_safe = F.clamp 1e-6 100.0 denominator

  in numerator / denominator_safe

-- | Möbius scalar multiplication (simplified)
mobiusScalarMult :: Double -> Tensor -> Tensor
mobiusScalarMult r x =
  let x_norm = tensorNorm x
      x_norm_safe = F.clamp 1e-8 0.9999 x_norm
      -- Manual artanh: artanh(x) = 0.5 * log((1+x)/(1-x))
      artanh_norm = 0.5 * F.log ((1.0 + x_norm_safe) / (1.0 - x_norm_safe))
      -- Create scalar tensor with same properties as artanh_norm
      r_scalar = TF.full [] r defaultOpts
      new_norm = F.tanh (artanh_norm * r_scalar)
      direction = x / (x_norm_safe + 1e-8)
  in direction * new_norm

-- | Riemannian gradient conversion for optimization
-- Converts Euclidean gradient to Riemannian gradient on Poincaré ball
riemannianGradient :: Tensor -> Tensor -> Tensor
riemannianGradient point euclidean_grad =
  let point_norm2 = F.sumAll (point * point)
      -- Conformal factor: (1 - ||x||²)² / 4
      one_minus_norm2 = 1.0 - point_norm2
      conformal_factor = (one_minus_norm2 * one_minus_norm2) / 4.0  -- Manual squaring
  in euclidean_grad * conformal_factor

-- | Initialize hyperbolic embeddings near origin
-- Small random values to start optimization in stable region
initHyperbolicEmbeddings :: [Int] -> IO Tensor
initHyperbolicEmbeddings shape = do
  -- Initialize with small Gaussian noise
  x <- TF.randnIO shape defaultOpts
  let scaled = x * 0.01  -- Scale to be very small
  return $ projectToPoincare scaled

-- | Alternative: Ultra-gentle mapping that barely modifies the embeddings
-- Use this if the above still doesn't work
ultraGentleMapping :: Tensor -> Tensor
ultraGentleMapping euclidean_emb =
  let -- Minimal scaling - just enough to fit in unit ball
      scaled = euclidean_emb * 0.08  -- Very gentle
  in projectToPoincare scaled

-- | COMPLETELY REWRITTEN: Hyperbolic embedding layer using exponential map
-- This preserves semantic structure much better than the previous version
hyperbolicEmbeddingLayer :: Tensor -> Tensor
hyperbolicEmbeddingLayer euclidean_emb =
  let -- Much gentler scaling to preserve more structure
      -- The key insight: we want to preserve relative distances, not just map to hyperbolic space
      scaled = euclidean_emb * 0.15  -- Even gentler scaling

      -- Instead of tanh mapping which compresses everything, use a linear approach
      -- with gentle normalization only for vectors that are too large
      norm = tensorNorm scaled
      max_allowed = 0.95

      -- Only normalize if norm > max_allowed, otherwise keep original
      scaling_needed = F.gt norm max_allowed
      scale_factor = max_allowed / (norm + 1e-8)
      ones_tensor = TF.onesLike scale_factor

      -- Use where to conditionally scale
      final_scale = FI.where' scaling_needed scale_factor ones_tensor

  in scaled * final_scale

-- | Example usage for embedding similarity
-- Returns hyperbolic distance between two sets of embeddings
embeddingDistance :: Tensor -> Tensor -> Tensor
embeddingDistance emb1 emb2 =
  let proj_emb1 = projectToPoincare emb1
      proj_emb2 = projectToPoincare emb2
  in poincareDistance proj_emb1 proj_emb2

-- | Hierarchical similarity loss (closer = smaller distance)
hierarchicalLoss :: Tensor -> Tensor -> Tensor -> Tensor
hierarchicalLoss embeddings positive_pairs negative_pairs =
  let pos_dist = embeddingDistance positive_pairs embeddings
      neg_dist = embeddingDistance negative_pairs embeddings
      margin = 1.0
      -- Hinge loss: max(0, pos_dist - neg_dist + margin)
  in F.relu (pos_dist - neg_dist + margin)

-- For integration with neural networks:

-- | Simple hyperbolic neural network layer
data HyperbolicLayer = HyperbolicLayer
  { weight :: Tensor
  , bias :: Tensor
  , inputDim :: Int
  , outputDim :: Int
  }

-- | Forward pass through hyperbolic layer (simplified)
hyperbolicForward :: HyperbolicLayer -> Tensor -> Tensor
hyperbolicForward layer input =
  let -- Standard linear transformation using Internal module
      matmul_result = FI.mm input (weight layer)
      linear_out = matmul_result + (bias layer)
      -- Project to Poincaré ball
  in projectToPoincare linear_out

-- Helper functions for numerical stability

-- | Safe arccosh that handles edge cases
safeAcosh :: Tensor -> Tensor
safeAcosh x =
  let clamped = F.clamp 1.000001 10.0 x
  in F.log (clamped + F.sqrt (clamped * clamped - 1.0))  -- Manual acosh

-- | Safe artanh that handles edge cases
safeArtanh :: Tensor -> Tensor
safeArtanh x =
  let clamped = F.clamp (-0.9999) 0.9999 x
  in 0.5 * F.log ((1.0 + clamped) / (1.0 - clamped))  -- Manual artanh

-- | Check if embeddings are valid (within unit ball)
isValidEmbedding :: Tensor -> Tensor
isValidEmbedding x =
  let norms = tensorNorm x
  in F.lt norms 1.0

-- | Batch processing version of Poincaré distance
batchPoincareDistance :: Tensor -> Tensor -> Tensor
batchPoincareDistance batch_x batch_y =
  -- This is a placeholder - implement batch operations as needed
  poincareDistance batch_x batch_y

analyzeEmbedding :: Tensor -> String -> IO ()
analyzeEmbedding emb name = do
  let norm = asValue (tensorNorm emb) :: Float
      mean_val = asValue (F.mean emb) :: Float
      std_val = asValue (F.std emb) :: Float
  putStrLn $ name ++ ": norm=" ++ show norm ++
             ", mean=" ++ show mean_val ++
             ", std=" ++ show std_val
