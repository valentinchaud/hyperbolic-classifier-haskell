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

-- | Project vectors to Poincaré ball (keep within unit ball)
-- This is the most critical function - prevents numerical issues
projectToPoincare :: Tensor -> Tensor
projectToPoincare x =
  let norms = tensorNorm x
      -- Only scale vectors that are >= 1.0, preserve smaller ones
      scaling_factor = 0.95 / (norms + 1e-8)  -- Scale to 0.95
      mask = F.ge norms 1.0  -- Scale only if norm >= 1.0
      ones_tensor = TF.onesLike norms
      scaling = FI.where' mask scaling_factor ones_tensor
  in x * scaling

poincareDistance :: Tensor -> Tensor -> Tensor
poincareDistance x y =
  -- For individual word embeddings, we manually compute the dot products
  let -- Compute element-wise products and sum them manually
      x_dot_x = F.sumAll (x * x)  -- ||x||²
      y_dot_y = F.sumAll (y * y)  -- ||y||²
      diff = x - y
      diff_dot_diff = F.sumAll (diff * diff)  -- ||x-y||²

      -- Poincaré distance formula: d = acosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
      numerator = 2.0 * diff_dot_diff
      denominator = (1.0 - x_dot_x) * (1.0 - y_dot_y)

      -- Add epsilon for numerical stability
      epsilon = 1e-8
      denominator_safe = F.clamp epsilon 1.0 denominator

      inner_term = 1.0 + numerator / denominator_safe
      clamped_inner = F.clamp 1.000001 10.0 inner_term
  in F.log (clamped_inner + F.sqrt (clamped_inner * clamped_inner - 1.0))

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

-- | Möbius addition (simplified version)
mobiusAdd :: Tensor -> Tensor -> Tensor
mobiusAdd u v =
  let u_norm2 = F.sumAll (u * u)
      v_norm2 = F.sumAll (v * v)
      uv_dot = F.sumAll (u * v)

      -- Numerator terms
      coeff1 = 1.0 + 2.0 * uv_dot + v_norm2
      coeff2 = 1.0 - u_norm2
      numerator = u * coeff1 + v * coeff2

      -- Denominator
      denominator = 1.0 + 2.0 * uv_dot + u_norm2 * v_norm2 + 1e-8

  in numerator / denominator

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

-- | Hyperbolic embedding layer - projects Euclidean embeddings to Poincaré ball
-- Uses a gentler normalization to preserve relative magnitudes
hyperbolicEmbeddingLayer :: Tensor -> Tensor
hyperbolicEmbeddingLayer euclidean_emb =
  let -- Scale down large embeddings more gently
      scaled = euclidean_emb * 0.1  -- Gentler scaling instead of tanh
  in projectToPoincare scaled

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
