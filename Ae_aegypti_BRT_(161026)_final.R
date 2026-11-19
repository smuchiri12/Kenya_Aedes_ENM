# =============================================================================
# COMPLETE PIPELINE – NO STANDARDISATION (RAW COVARIATES)
# =============================================================================
# CHANGES FROM ORIGINAL:
#   1. Added keep.fold.fit = TRUE to both gbm.step() calls
#   2. Replaced SI_ppc block with corrected out-of-fold Sørensen Index (SI)
#   3. Renamed SIppcs -> SIs, SI_ppc -> SI throughout
#   4. Added 95% CI rasters alongside SD rasters in annual predictions
#   5. Added 95% CI columns to performance summary tables
#   6. Updated plot labels SIppc -> SI
#   7. REMOVED TSS and F1 from all validation metrics — only AUC, SI, Boyce
# =============================================================================

library(raster)
library(dismo)
library(gbm)
library(blockCV)
library(sf)
library(dplyr)
library(ggplot2)
library(purrr)
library(stringr)
library(terra)
library(ecospat)

# ─────────────────────────────────────────────────────────────────────────────
# PART 0: CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

monthly_input_folder <- "E:/Ecological niche modelling/Covariates/Final covariates for extraction resampled 5km (120626)"
annual_output_base   <- "covariates_5km_v2"
kenya_shapefile_path <- "C:/Users/SMuchiri/OneDrive - Kemri Wellcome Trust/Kenya_Admin_0_2_Boundaries/Cleaned Admin 0 - 2 (290121)/KEN_Admin_0_(290121).shp"


# Species data (raw or cleaned) – adjust if needed
raw_rds      <- "RDS Data/Ae_aegypti_PA_100_5km_bias_with_neighbour_exclusion_with_covariates_collapsed_(240926)_for_model.rds"
cleaned_rds  <- "RDS Data/Ae_aegypti_PA_100_5km_bias_with_neighbour_exclusion_with_covariates_collapsed_(240926)_for_model_cl.rds"

years <- 2000:2025

annual_covariates <- c("animal_population", "log_human_population", "percent_rangeland", "percent_bare",
                       "percent_cropland", "percent_forest", "percent_wetland", "percent_urban", "percent_built_up")

time_varying_covariates <- list(
  lst_day        = list(method = "mean"),
  lst_night      = list(method = "mean"),
  precipitation  = list(method = "mean"),
  soilmoisture   = list(method = "mean"),
  aet            = list(method = "mean"),
  windspeed      = list(method = "mean"),
  evi            = list(method = "mean")
)

# # ---- SPECIES-SPECIFIC COVARIATES (EDIT FOR Ae. aegypti) ----
# #Only those listed below will be used in the model.

############covariates for 5km##############

species_covariates <- list(
  Ae_aegypti = list(
    static = c("elevation"),
    time_varying = c("percent_built_up", "percent_cropland",
                     "percent_rangeland",
                     "percent_urban", "percent_wetland",
                     "log_human_population",
                     "lst_night", "lst_day", "evi")
  )
)

# species_covariates <- list(
#   Ae_aegypti = list(
#     time_varying = c("percent_built_up",
#                      "percent_rangeland", 
#                      "percent_wetland", 
#                      "log_human_population",
#                      "lst_night", "evi")
#   )
# )

# ############covariates for 10km##########
# species_covariates <- list(
#   Ae_aegypti = list(
#     static = c("slope"),
#     time_varying = c("percent_bare", "percent_wetland","animal_population")
#   )
# )

############covariates for 20km##############

# species_covariates <- list(
#   Ae_aegypti = list(
#     static = c("slope","dist_road","flood"),
#     time_varying = c("percent_bare", "percent_wetland")
#   )
# )

# species_covariates <- list(
#   Ae_aegypti = list(
#     static = c("soil_ph", "elevation", "dist_water", "slope", "dist_river"),
#     time_varying = c("percent_bare", "percent_wetland", "percent_rangeland", "percent_cropland",
#                      "percent_urban", "animal_population")
#   )
# )

# ---- BRT settings ----
species_to_run     <- c("Ae_aegypti")
nberOfReplicates   <- 100
tree.complexity    <- 5
learning.rate      <- 0.01
bag.fraction       <- 0.80
n.folds            <- 5
max.trees          <- 10000
step.size          <- 5  #try with step.size = 5, 10, 50
tolerance          <- 0.001
family             <- "bernoulli"
spatial_block_size <- 300000
out_base           <- "BRT_results"
savingPlots        <- TRUE

# ---- Data cleaning ----
na_strategy          <- "remove_rows"
max_retries          <- 3
n_models_for_RI      <- 100
n_models_for_predictions <- 100
map_years            <- seq(2000, 2025, 5)
map_dpi              <- 300

name_mapping <- c(
  "Precipitation" = "precipitation",
  "Daytime_LST" = "lst_day",
  "Windspeed" = "windspeed",
  "Soil_moisture" = "soilmoisture",
  "Actual_evapotranspiration" = "aet",
  "EVI" = "evi",
  "Nighttime_LST" = "lst_night",
  "Percent_builtup" = "percent_built_up",
  "Percent_wetlands" = "percent_wetland",
  "Percent_bareareas" = "percent_bare",
  "Percent_croplands" = "percent_cropland",
  "Percent_forest" = "percent_forest",
  "Percent_rangelands" = "percent_rangeland",
  "Animal_population_density" = "animal_population",
  "Percent_urban" = "percent_urban",
  "Slope" = "slope",
  "Soil_pH" = "soil_ph",
  "Distance_to_rivers" = "dist_river",
  "Distance_to_roads" = "dist_road",
  "Distance_to_waterbodies" = "dist_water",
  "Elevation" = "elevation",
  "Flooded_extents" = "flood",
  "Strahler_number" = "strahler_number",
  "Human_population_density" = "log_human_population"
)
old_to_new <- name_mapping

target_crs <- CRS("+proj=longlat +datum=WGS84 +no_defs")
kenya_ext <- extent(33.9095878600000447, 41.9060214590000442, 
                    -4.7204170229999249, 5.4141235350000443)
target_res <- 0.04166
template_raster <- raster(ext = kenya_ext, resolution = target_res, crs = target_crs)
values(template_raster) <- NA

# =============================================================================
# STEP 0: COPY ANNUAL RASTERS
# =============================================================================
cat("\n╔════════════════════════════════════════════════════════════╗\n")
cat("  STEP 0: Copying annual rasters (year‑specific)\n")
cat("╚════════════════════════════════════════════════════════════╝\n")

for (year in years) {
  dir.create(paste0(annual_output_base, "/", year), recursive = TRUE, showWarnings = FALSE)
}
dir.create("covariates/static", recursive = TRUE, showWarnings = FALSE)

for (cov in annual_covariates) {
  cat("\nProcessing:", cov, "\n")
  for (year in years) {
    dest_file <- paste0(annual_output_base, "/", year, "/", cov, "_", year, ".tif")
    if (file.exists(dest_file)) {
      cat("  Year", year, "— already exists\n")
      next
    }
    src_file <- paste0(monthly_input_folder, "/", cov, "_", year, ".tif")
    if (!file.exists(src_file)) {
      src_file_alt <- paste0(monthly_input_folder, "/", cov, year, ".tif")
      if (!file.exists(src_file_alt)) {
        warning("Source file not found for ", cov, " year ", year, ". Skipping.")
        next
      } else {
        src_file <- src_file_alt
      }
    }
    src <- raster(src_file)
    aligned <- resample(src, template_raster, method = "bilinear")
    writeRaster(aligned, dest_file, format = "GTiff", overwrite = TRUE, options = "COMPRESS=LZW")
    cat("  Created:", basename(dest_file), "\n")
  }
}
cat("\n✓ Annual rasters copied and aligned.\n\n")

# =============================================================================
# STEP 1: AGGREGATE MONTHLY RASTERS TO ANNUAL
# =============================================================================
cat("\n╔════════════════════════════════════════════════════════════╗\n")
cat("  STEP 1: Monthly to Annual Aggregation (fast, terra)\n")
cat("╚════════════════════════════════════════════════════════════╝\n")

for (cov_name in names(time_varying_covariates)) {
  method <- time_varying_covariates[[cov_name]]$method
  cat("\n────────────────────────────────────────\n")
  cat("Covariate:", cov_name, "| Method:", toupper(method), "\n")
  
  for (year in years) {
    output_file <- paste0(annual_output_base, "/", year, "/", cov_name, "_", year, ".tif")
    if (file.exists(output_file)) {
      cat("  Year", year, "— already exists\n")
      next
    }
    month_files <- list.files(monthly_input_folder,
                              pattern = paste0(cov_name, "_", year, "_\\d{1,2}\\.tif$"),
                              full.names = TRUE)
    if (length(month_files) == 0) {
      cat("  Year", year, "— no monthly files\n")
      next
    }
    cat("  Year", year, "— found", length(month_files), "months\n")
    
    r <- rast(month_files)
    if (cov_name %in% c("lst_day", "lst_night")) {
      if (method == "mean") {
        annual_terra <- mean(r, na.rm = TRUE) - 273.15
      } else {
        annual_terra <- sum(r, na.rm = TRUE) - 273.15 * nlyr(r)
      }
    } else {
      if (method == "mean") {
        annual_terra <- mean(r, na.rm = TRUE)
      } else {
        annual_terra <- sum(r, na.rm = TRUE)
      }
    }
    annual_raster <- raster(annual_terra)
    aligned <- resample(annual_raster, template_raster, method = "bilinear")
    writeRaster(aligned, output_file, format = "GTiff", overwrite = TRUE, options = "COMPRESS=LZW")
    cat("    ✓ Saved (aligned)\n")
    rm(r, annual_terra, annual_raster, aligned); gc()
  }
}
cat("\n✓ Monthly aggregation complete.\n\n")

# =============================================================================
# STEP 2: DATA CLEANING & RENAMING
# =============================================================================
cat("\n╔════════════════════════════════════════════════════════════╗\n")
cat("  STEP 2: Data Cleaning & Renaming\n")
cat("╚════════════════════════════════════════════════════════════╝\n")
if (file.exists(cleaned_rds)) {
  cat("Removing old cleaned RDS to force recreation with current covariates.\n")
  file.remove(cleaned_rds)
}

if (!file.exists(cleaned_rds)) {
  cat("Loading raw data...\n")
  train_raw <- readRDS(raw_rds)
  if (!is.list(train_raw)) train_raw <- split(train_raw, train_raw$df_id)
  cat("Replicates:", length(train_raw), "\n")
  
  sample_names <- names(train_raw[[1]])
  cols_to_rename <- intersect(names(old_to_new), sample_names)
  cat("Renaming columns:", paste(cols_to_rename, collapse = ", "), "\n")
  
  train_renamed <- lapply(train_raw, function(df) {
    for (old in cols_to_rename) {
      new <- old_to_new[old]
      if (old %in% colnames(df)) colnames(df)[colnames(df) == old] <- new
    }
    if ("Longitude" %in% colnames(df) && !("x" %in% colnames(df))) 
      colnames(df)[colnames(df) == "Longitude"] <- "x"
    if ("Latitude" %in% colnames(df) && !("y" %in% colnames(df))) 
      colnames(df)[colnames(df) == "Latitude"] <- "y"
    if (!("response" %in% colnames(df)) && "presence" %in% colnames(df)) 
      colnames(df)[colnames(df) == "presence"] <- "response"
    for (col in c("x", "y", "year", "month", "response", old_to_new)) {
      if (col %in% colnames(df) && !is.numeric(df[[col]])) 
        df[[col]] <- as.numeric(as.character(df[[col]]))
    }
    return(df)
  })
  
  all_covs_raw <- unique(c(unlist(species_covariates[[species_to_run[1]]]), names(old_to_new)))
  essential <- c("x", "y", "response", "year", "month")
  cols_keep <- unique(c(essential, all_covs_raw))
  cols_keep <- cols_keep[cols_keep %in% names(train_renamed[[1]])]
  
  train_clean <- lapply(train_renamed, function(df) {
    df <- df[, cols_keep, drop = FALSE]
    if (na_strategy == "remove_rows") {
      na_rows <- apply(df[, -which(names(df) %in% c("x", "y"))], 1, function(r) any(is.na(r)))
      df <- df[!na_rows, ]
    }
    return(df)
  })
  saveRDS(train_clean, cleaned_rds)
  cat("Cleaned data saved (raw, no standardisation).\n")
} else {
  cat("Cleaned data already exists – loading.\n")
  train_clean <- readRDS(cleaned_rds)
}

# =============================================================================
# STEP 3: SELECT SPECIES-SPECIFIC COVARIATES
# =============================================================================
cat("\n╔════════════════════════════════════════════════════════════╗\n")
cat("  STEP 3: Covariate Selection (raw values)\n")
cat("╚════════════════════════════════════════════════════════════╝\n")

sp_covs <- species_covariates[[species_to_run[1]]]
all_covs <- c(sp_covs$static, sp_covs$time_varying)
cat("Selected covariates for modelling (", length(all_covs), "): ", paste(all_covs, collapse = ", "), "\n")

train_subset <- lapply(train_clean, function(df) {
  keep <- c("x", "y", "response", "year", "month", all_covs)
  keep <- keep[keep %in% names(df)]
  df[, keep, drop = FALSE]
})

train_sets <- train_subset
response_col <- "response"
cov_cols <- all_covs

cat("Final training data: ", length(train_sets), "replicates, ", length(cov_cols), "covariates (raw)\n")

# =============================================================================
# STEP 4: BRT MODELLING, EVALUATION, PREDICTIONS & MAPS
# =============================================================================
cat("\n╔════════════════════════════════════════════════════════════╗\n")
cat("  STEP 4: BRT Modelling & Predictions (raw covariates)\n")
cat("╚════════════════════════════════════════════════════════════╝\n")

kenya_boundary <- st_read(kenya_shapefile_path, quiet = TRUE)
kenya_boundary <- st_transform(kenya_boundary, crs = 4326)
kenya_sp <- as(kenya_boundary, "Spatial")

for (sp in species_to_run) {
  for (subdir in c("models", "predictions/mean", "predictions/stdv", "performance",
                   "relative_influence", "response_curves", "diagnostics", "maps")) {
    dir.create(paste0(out_base, "/", sp, "/", subdir), recursive = TRUE, showWarnings = FALSE)
  }
}

for (species_name in species_to_run) {
  cat("\n────────────────────────────────────────\n")
  cat("Processing:", species_name, "\n")
  
  ccv_path <- paste0(out_base, "/", species_name, "/models/BRT_models_CCV.rds")
  scv_path <- paste0(out_base, "/", species_name, "/models/BRT_models_SCV.rds")
  auc_path <- paste0(out_base, "/", species_name, "/performance/AUC_values.csv")
  
  if (!file.exists(ccv_path) || !file.exists(scv_path)) {
    brt_models_ccv <- list()
    brt_models_scv <- list()
    AUCs <- data.frame(CCV_AUC = rep(NA, nberOfReplicates), 
                       SCV_AUC = rep(NA, nberOfReplicates))
    
    for (i in 1:nberOfReplicates) {
      cat("  Replicate", i, "/", nberOfReplicates, "... ")
      data <- as.data.frame(train_sets[[i]])
      data$response <- data[[response_col]]
      for (col in cov_cols) {
        if (!is.numeric(data[[col]])) data[[col]] <- as.numeric(as.character(data[[col]]))
      }
      
      model_ccv <- NULL
      for (retry in 1:max_retries) {
        tryCatch({
          model_ccv <- gbm.step(
            data = data, gbm.x = cov_cols, gbm.y = "response",
            tree.complexity = tree.complexity, learning.rate = learning.rate,
            bag.fraction = bag.fraction, n.folds = n.folds, prev.stratify = TRUE,
            family = family, n.trees = 50, step.size = step.size, max.trees = max.trees,
            tolerance.method = "auto", tolerance = tolerance, plot.main = FALSE,
            verbose = FALSE, silent = TRUE,
            keep.fold.fit = TRUE
          )
        }, error = function(e) {})
        if (!is.null(model_ccv)) break
      }
      if (is.null(model_ccv)) { cat("CCV failed\n"); next }
      brt_models_ccv[[i]] <- model_ccv
      AUCs$CCV_AUC[i] <- model_ccv$cv.statistics$discrimination.mean
      
      model_scv <- NULL
      for (retry in 1:max_retries) {
        tryCatch({
          spdf <- SpatialPointsDataFrame(coords = data[, c("x", "y")], data = data,
                                         proj4string = CRS("+proj=longlat +datum=WGS84"))
          set.seed(i)
          myblocks <- cv_spatial(x = spdf, column = "response", k = n.folds,
                                 size = spatial_block_size, selection = "random",
                                 iteration = 100)
          model_scv <- gbm.step(
            data = data, gbm.x = cov_cols, gbm.y = "response",
            fold.vector = myblocks$folds_ids, tree.complexity = tree.complexity,
            learning.rate = learning.rate, bag.fraction = bag.fraction,
            n.folds = n.folds, prev.stratify = TRUE, family = family,
            n.trees = 50, step.size = step.size, max.trees = max.trees,
            tolerance.method = "auto", tolerance = tolerance, plot.main = FALSE,
            verbose = FALSE, silent = TRUE,
            keep.fold.fit = TRUE, keep.fold.vector = TRUE
          )
        }, error = function(e) {})
        if (!is.null(model_scv)) break
      }
      if (is.null(model_scv)) {
        cat("SCV failed (CCV AUC =", round(AUCs$CCV_AUC[i],3), ")\n")
      } else {
        brt_models_scv[[i]] <- model_scv
        AUCs$SCV_AUC[i] <- model_scv$cv.statistics$discrimination.mean
        cat("CCV AUC =", round(AUCs$CCV_AUC[i],3), 
            "| SCV AUC =", round(AUCs$SCV_AUC[i],3), "\n")
      }
      if (i %% 10 == 0) {
        saveRDS(brt_models_ccv, paste0(ccv_path, ".checkpoint"))
        saveRDS(brt_models_scv, paste0(scv_path, ".checkpoint"))
      }
    }
    saveRDS(brt_models_ccv, ccv_path)
    saveRDS(brt_models_scv, scv_path)
    write.csv(AUCs, auc_path, row.names = FALSE)
  } else {
    cat("Models already exist – loading\n")
    brt_models_scv <- readRDS(scv_path)
    AUCs <- read.csv(auc_path)
  }
  
  valid_idx <- which(!is.na(AUCs$SCV_AUC))
  if (length(valid_idx) == 0) {
    cat("No valid SCV models – skipping further steps\n")
    next
  }
  valid_models <- brt_models_scv[valid_idx]
  valid_aucs <- AUCs$SCV_AUC[valid_idx]
  valid_order <- order(valid_aucs, decreasing = TRUE)
  valid_models <- valid_models[valid_order]
  
  cat("Successful SCV models:", length(valid_models), "\n")
  cat("Mean SCV AUC:", round(mean(valid_aucs, na.rm = TRUE), 4), "\n")
  
  # =========================================================================
  # Sørensen Index (SI) on out-of-fold predictions
  # =========================================================================
  cat("\nComputing Sørensen Index (out-of-fold predictions)...\n")
  si_path <- paste0(out_base, "/", species_name, "/performance/SI_internal.csv")
  if (!file.exists(si_path)) {
    SIs <- data.frame(SI = rep(NA, nberOfReplicates),
                      threshold = rep(NA, nberOfReplicates))
    thresholds <- seq(0, 1, 0.01)
    for (idx in 1:length(valid_models)) {
      if (idx %% 10 == 0) cat("  Model", idx, "/", length(valid_models), "\n")
      model <- valid_models[[idx]]
      responses <- model$gbm.call$dataframe[, model$gbm.call$gbm.y]
      foldFit <- model$fold.fit
      
      if (is.null(foldFit) || length(foldFit) != length(responses)) {
        cat("  Replicate", idx, ": fold.fit missing or length mismatch — skipping\n")
        next
      }
      
      prediction <- stats::plogis(foldFit)
      
      P <- sum(responses == 1); A <- sum(responses == 0); prev <- P/(P+A)
      x <- (P/A)*((1-prev)/prev)
      TP <- sapply(thresholds, function(th) sum((responses == 1) & (prediction >= th)))
      FP <- sapply(thresholds, function(th) sum((responses == 0) & (prediction >= th)))
      FN <- sapply(thresholds, function(th) sum((responses == 1) & (prediction < th)))
      si_values <- (2*TP)/((2*TP) + (x*FP) + FN)
      best_idx <- which.max(si_values)
      original_pos <- valid_idx[valid_order][idx]
      SIs$SI[original_pos] <- round(si_values[best_idx], 3)
      SIs$threshold[original_pos] <- thresholds[best_idx]
    }
    write.csv(SIs, si_path, row.names = FALSE)
  } else {
    SIs <- read.csv(si_path)
  }
  
  mean_SI <- round(mean(SIs$SI, na.rm = TRUE), 3)
  ci_SI <- round(quantile(SIs$SI, c(0.025, 0.975), na.rm = TRUE), 3)
  cat("Mean SI =", mean_SI,
      "| 2.5th-97.5th percentiles = [", ci_SI[1], ",", ci_SI[2], "]\n")
  
  # =========================================================================
  # BOYCE INDEX (Internal / Training data)
  # =========================================================================
  cat("\nComputing Boyce Index for internal evaluation...\n")
  boyce_path <- paste0(out_base, "/", species_name, "/performance/Boyce_internal.csv")
  
  if (!file.exists(boyce_path)) {
    Boyce_vals <- data.frame(replicate = 1:nberOfReplicates, Boyce = NA)
    
    for (i in valid_idx) {
      model <- brt_models_scv[[i]]
      if (is.null(model)) next
      
      resp <- model$gbm.call$dataframe[[model$gbm.call$gbm.y]]
      ff   <- model$fold.fit
      fv   <- model$fold.vector
      if (is.null(ff) || is.null(fv) ||
          length(ff) != length(resp) || length(fv) != length(resp)) {
        cat("  Replicate", i, ": fold.fit or fold.vector missing - skipping\n")
        next
      }
      pred <- stats::plogis(ff)                      # out-of-fold predictions
      
      boyce_k <- sapply(sort(unique(fv)), function(k) {
        idx    <- fv == k
        pres_k <- pred[idx & resp == 1]
        if (sum(idx) < 2 || length(pres_k) < 2) return(NA_real_)
        r <- tryCatch(ecospat.boyce(fit = pred[idx], obs = pres_k, PEplot = FALSE)$cor,
                      error = function(e) NA)
        if (length(r) == 1 && is.finite(r)) r else NA_real_
      })
      Boyce_vals$Boyce[i] <- mean(boyce_k, na.rm = TRUE)   # mean of the 5 per-fold CBI values
      
      if (i %% 10 == 0) cat("  Processed", i, "replicates\n")
    }
    write.csv(Boyce_vals, boyce_path, row.names = FALSE)
  } else {
    Boyce_vals <- read.csv(boyce_path)
  }
  
  # =========================================================================
  # INTERNAL METRICS: AUC, SI, Boyce only (TSS and F1 REMOVED)
  # =========================================================================
  cat("\nAssembling internal metrics (AUC, SI, Boyce)...\n")
  
  internal_metrics <- data.frame(
    replicate = 1:nberOfReplicates,
    AUC = NA,
    SI = NA,
    Boyce = NA
  )
  
  for (i in valid_idx) {
    internal_metrics$AUC[i]   <- AUCs$SCV_AUC[i]
    internal_metrics$SI[i]    <- SIs$SI[i]
    internal_metrics$Boyce[i] <- Boyce_vals$Boyce[i]
  }
  
  write.csv(internal_metrics, 
            paste0(out_base, "/", species_name, "/performance/internal_metrics.csv"), 
            row.names = FALSE)
  
  # ---- Internal boxplot (AUC, SI, Boyce only) ----
  library(ggplot2)
  library(tidyr)
  
  internal_long <- internal_metrics %>%
    dplyr::select(replicate, AUC, SI, Boyce) %>%
    pivot_longer(cols = c(AUC, SI, Boyce), 
                 names_to = "Metric", values_to = "Value")
  
  internal_long$Metric <- factor(internal_long$Metric, 
                                 levels = c("AUC", "SI", "Boyce"))
  
  p_internal <- ggplot(internal_long, aes(x = Metric, y = Value, fill = Metric)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, size = 0.8, alpha = 0.4) +
    labs(title = paste0(species_name, " – Internal Performance (Training)"),
         y = "Value", x = "") +
    ylim(0, 1) +
    theme_minimal() +
    theme(legend.position = "none")
  
  ggsave(paste0(out_base, "/", species_name, "/performance/internal_metrics_boxplot.pdf"),
         p_internal, width = 6, height = 5, dpi = 300)
  cat("  Internal metrics boxplot saved.\n")
  
  # =========================================================================
  # INTERNAL CURVES: ROC & SI – Mean ± 95% CI
  # =========================================================================
  cat("\nComputing internal ROC and SI curves across replicates...\n")
  
  thresholds <- seq(0, 1, by = 0.005)
  n_curves <- length(valid_idx)
  
  roc_train_tpr <- matrix(NA, nrow = length(thresholds), ncol = n_curves)
  roc_train_fpr <- matrix(NA, nrow = length(thresholds), ncol = n_curves)
  si_train <- matrix(NA, nrow = length(thresholds), ncol = n_curves)
  
  for (idx in 1:n_curves) {
    i <- valid_idx[idx]
    model <- brt_models_scv[[i]]
    if (is.null(model)) next
    
    responses <- model$gbm.call$dataframe[[model$gbm.call$gbm.y]]
    if (is.null(model$fold.fit) || length(model$fold.fit) != length(responses)) next
    pred <- stats::plogis(model$fold.fit)
    
    P <- sum(responses == 1); A <- sum(responses == 0)
    prev <- P / (P + A)
    x <- (P / A) * ((1 - prev) / prev)
    
    for (t in seq_along(thresholds)) {
      th <- thresholds[t]
      tp <- sum(responses == 1 & pred >= th)
      fp <- sum(responses == 0 & pred >= th)
      fn <- sum(responses == 1 & pred < th)
      tn <- sum(responses == 0 & pred < th)
      roc_train_tpr[t, idx] <- ifelse((tp + fn) == 0, NA, tp / (tp + fn))
      roc_train_fpr[t, idx] <- ifelse((fp + tn) == 0, NA, fp / (fp + tn))
      si <- ifelse((2*tp + x*fp + fn) == 0, 0, (2*tp) / (2*tp + x*fp + fn))
      si_train[t, idx] <- si
    }
    if (idx %% 10 == 0) cat("  Processed", idx, "replicates (internal)\n")
  }
  
  mean_roc_train_tpr <- rowMeans(roc_train_tpr, na.rm = TRUE)
  mean_roc_train_fpr <- rowMeans(roc_train_fpr, na.rm = TRUE)
  ci_roc_train_tpr_low <- apply(roc_train_tpr, 1, quantile, 0.025, na.rm = TRUE)
  ci_roc_train_tpr_high <- apply(roc_train_tpr, 1, quantile, 0.975, na.rm = TRUE)
  mean_si_train <- rowMeans(si_train, na.rm = TRUE)
  ci_si_train_low <- apply(si_train, 1, quantile, 0.025, na.rm = TRUE)
  ci_si_train_high <- apply(si_train, 1, quantile, 0.975, na.rm = TRUE)
  
  # ---- Plot internal curves ----
  roc_internal_df <- data.frame(
    FPR = mean_roc_train_fpr,
    TPR = mean_roc_train_tpr,
    CI_low = ci_roc_train_tpr_low,
    CI_high = ci_roc_train_tpr_high
  )
  roc_internal_df <- na.omit(roc_internal_df)
  
  p_roc_int <- ggplot(roc_internal_df, aes(x = FPR, y = TPR)) +
    geom_ribbon(aes(ymin = CI_low, ymax = CI_high), fill = "#2E86AB", alpha = 0.2) +
    geom_line(color = "#2E86AB", size = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
    labs(x = "1 - Specificity (False Positive Rate)", 
         y = "Sensitivity (True Positive Rate)",
         title = paste0(species_name, " – Internal ROC Curve"),
         subtitle = paste0("Mean ± 95% CI (", n_curves, " replicates)")) +
    coord_equal() +
    theme_minimal()
  
  ggsave(paste0(out_base, "/", species_name, "/performance/ROC_internal.pdf"),
         p_roc_int, width = 6, height = 6, dpi = 300)
  
  si_internal_df <- data.frame(
    Threshold = thresholds,
    SI = mean_si_train,
    CI_low = ci_si_train_low,
    CI_high = ci_si_train_high
  )
  si_internal_df <- na.omit(si_internal_df)
  
  p_si_int <- ggplot(si_internal_df, aes(x = Threshold, y = SI)) +
    geom_ribbon(aes(ymin = CI_low, ymax = CI_high), fill = "#2E86AB", alpha = 0.2) +
    geom_line(color = "#2E86AB", size = 1.2) +
    labs(x = "Threshold", 
         y = "Sørensen Index (SI)",
         title = paste0(species_name, " – Internal SI Curve"),
         subtitle = paste0("Mean ± 95% CI (", n_curves, " replicates)")) +
    ylim(0, 1) +
    theme_minimal()
  
  ggsave(paste0(out_base, "/", species_name, "/performance/SI_internal.pdf"),
         p_si_int, width = 6, height = 6, dpi = 300)
  
  cat("  Internal ROC and SI curves saved.\n")
  
  # ---- Relative Influence ----
  ri_path <- paste0(out_base, "/", species_name, "/relative_influence/RI_values.csv")
  if (!file.exists(ri_path)) {
    cat("\nComputing relative influence...\n")
    ri_matrix <- matrix(0, nrow = length(cov_cols), ncol = length(valid_models),
                        dimnames = list(cov_cols, NULL))
    for (idx in 1:length(valid_models)) {
      ri_summary <- summary(valid_models[[idx]], plotit = FALSE)
      if (nrow(ri_summary) > 0) {
        for (i in 1:nrow(ri_summary)) {
          cov <- ri_summary$var[i]
          if (cov %in% cov_cols) {
            ri_matrix[cov, idx] <- ri_summary$rel.inf[i]
          }
        }
      }
    }
    ri_results <- data.frame(
      covariate = cov_cols,
      mean_RI = round(rowMeans(ri_matrix), 2),
      sd_RI = round(apply(ri_matrix, 1, sd), 2)
    ) %>% arrange(desc(mean_RI))
    write.csv(ri_results, ri_path, row.names = FALSE)
    print(ri_results)
  } else {
    ri_results <- read.csv(ri_path)
  }
  
  # # ---- Response Curves (raw covariates) ----
  # rc_path <- paste0(out_base, "/", species_name, "/response_curves/response_curves.pdf")
  # if (savingPlots && !file.exists(rc_path) && length(valid_models) > 0) {
  #   
  #   cat("\n── Response curves (all", length(valid_models), "models) ──\n")
  #   cat("  Computing covariate ranges across all 100 dataframes...\n")
  #   
  #   env_vals <- lapply(cov_cols, function(cov) {
  #     vals <- unlist(lapply(train_sets, function(df) {
  #       if (cov %in% names(df)) df[[cov]] else NULL
  #     }))
  #     vals <- vals[!is.na(vals) & is.finite(vals)]
  #     if (length(vals) == 0) {
  #       c(min = NA_real_, max = NA_real_, med = NA_real_)
  #     } else {
  #       c(min = min(vals, na.rm = TRUE),
  #         max = max(vals, na.rm = TRUE),
  #         med = median(vals, na.rm = TRUE))
  #     }
  #   })
  #   names(env_vals) <- cov_cols
  #   
  #   valid_covs <- cov_cols[sapply(cov_cols, function(cov) {
  #     !is.na(env_vals[[cov]]["min"]) && !is.na(env_vals[[cov]]["max"]) &&
  #       is.finite(env_vals[[cov]]["min"]) && is.finite(env_vals[[cov]]["max"]) &&
  #       env_vals[[cov]]["min"] < env_vals[[cov]]["max"]
  #   })]
  #   
  #   if (length(valid_covs) == 0) {
  #     cat("  No valid covariates for response curves. Skipping.\n")
  #   } else {
  #     cat("  Using", length(valid_covs), "covariates\n")
  #     
  #     ri_list <- lapply(valid_models, function(model) {
  #       tryCatch({
  #         ri <- summary(model, plotit = FALSE)
  #         setNames(ri$rel.inf, ri$var)
  #       }, error = function(e) NULL)
  #     })
  #     ri_list <- Filter(Negate(is.null), ri_list)
  #     
  #     ri_mean <- sapply(valid_covs, function(cov) {
  #       vals <- sapply(ri_list, function(ri) if (cov %in% names(ri)) ri[cov] else 0)
  #       round(mean(vals, na.rm = TRUE), 1)
  #     })
  #     ri_min <- sapply(valid_covs, function(cov) {
  #       vals <- sapply(ri_list, function(ri) if (cov %in% names(ri)) ri[cov] else 0)
  #       round(min(vals, na.rm = TRUE), 1)
  #     })
  #     ri_max <- sapply(valid_covs, function(cov) {
  #       vals <- sapply(ri_list, function(ri) if (cov %in% names(ri)) ri[cov] else 0)
  #       round(max(vals, na.rm = TRUE), 1)
  #     })
  #     
  #     minY <- 1; maxY <- 0
  #     for (cov in valid_covs) {
  #       val_range <- seq(env_vals[[cov]]["min"], env_vals[[cov]]["max"], length.out = 100)
  #       df_rc <- as.data.frame(matrix(rep(env_vals[[cov]]["med"], 100),
  #                                     nrow = 100, ncol = length(cov_cols), byrow = TRUE))
  #       colnames(df_rc) <- cov_cols
  #       df_rc[, cov] <- val_range
  #       for (model in valid_models) {
  #         tryCatch({
  #           proj <- predict.gbm(model, df_rc, model$gbm.call$best.trees, type = "response")
  #           if (min(proj, na.rm = TRUE) < minY) minY <- min(proj, na.rm = TRUE)
  #           if (max(proj, na.rm = TRUE) > maxY) maxY <- max(proj, na.rm = TRUE)
  #         }, error = function(e) {})
  #       }
  #     }
  #     y_top <- maxY + ((maxY - minY) / 3)
  #     cat("  Y-axis range:", round(minY, 3), "to", round(maxY, 3), "\n")
  #     
  #     col_lines <- rgb(222, 67, 39, 30, maxColorValue = 255)
  #     col_mean  <- rgb(180, 40, 20, 255, maxColorValue = 255)
  #     col_range <- rgb(222, 67, 39, 40, maxColorValue = 255)
  #     
  #     n_cols <- 5
  #     n_rows <- ceiling(length(valid_covs) / n_cols)
  #     
  #     pdf(rc_path, width = n_cols * 1.5, height = n_rows * 1.5)
  #     par(mfrow = c(n_rows, n_cols), oma = c(0.5, 1.2, 0.8, 0.3),
  #         mar = c(2.0, 0.8, 0.3, 0.3), lwd = 0.2, col = "gray30")
  #     
  #     for (cov in valid_covs) {
  #       val_range <- seq(env_vals[[cov]]["min"], env_vals[[cov]]["max"], length.out = 100)
  #       df_rc <- as.data.frame(matrix(rep(env_vals[[cov]]["med"], 100),
  #                                     nrow = 100, ncol = length(cov_cols), byrow = TRUE))
  #       colnames(df_rc) <- cov_cols
  #       df_rc[, cov] <- val_range
  #       
  #       all_preds <- matrix(NA, nrow = 100, ncol = length(valid_models))
  #       for (j in seq_along(valid_models)) {
  #         tryCatch({
  #           all_preds[, j] <- predict.gbm(valid_models[[j]], df_rc,
  #                                         valid_models[[j]]$gbm.call$best.trees,
  #                                         type = "response")
  #         }, error = function(e) {})
  #       }
  #       all_preds <- all_preds[, colSums(!is.na(all_preds)) == 100, drop = FALSE]
  #       if (ncol(all_preds) == 0) {
  #         plot.new(); title(xlab = cov); box(); next
  #       }
  #       pred_mean <- rowMeans(all_preds, na.rm = TRUE)
  #       pred_min  <- apply(all_preds, 1, min, na.rm = TRUE)
  #       pred_max  <- apply(all_preds, 1, max, na.rm = TRUE)
  #       
  #       plot(val_range, pred_mean, col = NA, ann = FALSE, axes = FALSE, type = "n",
  #            xlim = c(env_vals[[cov]]["min"], env_vals[[cov]]["max"]),
  #            ylim = c(minY, y_top))
  #       polygon(c(val_range, rev(val_range)), c(pred_min, rev(pred_max)),
  #               col = col_range, border = NA)
  #       for (j in 1:ncol(all_preds)) lines(val_range, all_preds[, j], col = col_lines, lwd = 0.1)
  #       lines(val_range, pred_mean, col = col_mean, lwd = 1.5)
  #       
  #       axis(side = 1, lwd.tick = 0.2, cex.axis = 0.6, lwd = 0,
  #            tck = -0.04, col.axis = "gray30", mgp = c(0, 0.12, 0))
  #       axis(side = 2, lwd.tick = 0.2, cex.axis = 0.6, lwd = 0,
  #            tck = -0.04, col.axis = "gray30", mgp = c(0, 0.25, 0),
  #            at = seq(0, 1, 0.2))
  #       title(xlab = cov, cex.lab = 0.7, mgp = c(1.0, 0, 0), col.lab = "gray30")
  #       ri_val <- paste0(ri_mean[cov],"% [", ri_min[cov], "-", ri_max[cov], "]")
  #       mtext(paste0("RI = ", ri_val), side = 3, line = -1.2, cex = 0.45, col = "gray30")
  #       box(lwd = 0.3, col = "gray30")
  #     }
  #     dev.off()
  #     cat("  ✓ Response curves saved\n")
  #   }
  # }
  # 
  # # ---- Annual Predictions (raw rasters) ----
  # cat("\nGenerating annual predictions (raw covariates)...\n")
  # n_pred <- min(n_models_for_predictions, length(valid_models))
  # pred_models <- valid_models[1:n_pred]
  # 
  # static_rasters <- list()
  # for (cov in sp_covs$static) {
  #   path <- paste0("covariates/static/", cov, ".tif")
  #   if (file.exists(path)) {
  #     r <- raster(path)
  #     r <- resample(r, template_raster, method = "bilinear")
  #     static_rasters[[cov]] <- r
  #   } else {
  #     cat("Warning: static raster missing:", path, "\n")
  #   }
  # }
  # 
  # for (year in years) {
  #   mean_path <- paste0(out_base, "/", species_name, "/predictions/mean/",
  #                       species_name, "_", year, "_mean.tif")
  #   if (file.exists(mean_path)) next
  #   cat("  Year", year, "\n")
  #   
  #   tv_rasters <- list()
  #   for (cov in sp_covs$time_varying) {
  #     path <- paste0(annual_output_base, "/", year, "/", cov, "_", year, ".tif")
  #     if (file.exists(path)) {
  #       r <- raster(path)
  #       r <- resample(r, template_raster, method = "bilinear")
  #       tv_rasters[[cov]] <- r
  #     } else {
  #       cat("    Missing:", path, "\n")
  #     }
  #   }
  #   if (length(tv_rasters) != length(sp_covs$time_varying)) {
  #     cat("    Missing some time‑varying rasters\n")
  #     next
  #   }
  #   
  #   all_rasters <- c(static_rasters, tv_rasters)
  #   stack_all <- stack(all_rasters)
  #   names(stack_all) <- c(names(static_rasters), names(tv_rasters))
  #   
  #   df_full <- as.data.frame(stack_all)
  #   valid_idx_cells <- which(complete.cases(df_full))
  #   if (length(valid_idx_cells) == 0) { cat("    No valid cells\n"); next }
  #   df_valid <- df_full[valid_idx_cells, , drop = FALSE]
  #   
  #   projections <- list()
  #   for (j in 1:length(pred_models)) {
  #     model <- pred_models[[j]]
  #     model_vars <- model$gbm.call$gbm.x
  #     avail <- intersect(model_vars, colnames(df_valid))
  #     if (length(avail) == 0) next
  #     pred <- tryCatch({
  #       predict.gbm(model, df_valid[, avail, drop = FALSE],
  #                   model$gbm.call$best.trees, type = "response")
  #     }, error = function(e) NULL)
  #     if (is.null(pred)) next
  #     rast <- stack_all[[1]]
  #     rast[] <- NA
  #     rast[valid_idx_cells] <- pred
  #     projections[[length(projections)+1]] <- rast
  #   }
  #   
  #   # >>> CHANGED: save mean, SD, and 95% CI rasters
  #   if (length(projections) > 0) {
  #     proj_stack <- stack(projections)
  #     mean_proj <- calc(proj_stack, fun = mean, na.rm = TRUE)
  #     writeRaster(mean_proj, mean_path, format = "GTiff", overwrite = TRUE,
  #                 options = "COMPRESS=LZW")
  #     
  #     if (length(projections) >= 2) {
  #       # --- SD raster ---
  #       sd_path <- paste0(out_base, "/", species_name, "/predictions/stdv/",
  #                         species_name, "_", year, "_stdv.tif")
  #       sd_proj <- calc(proj_stack, fun = sd, na.rm = TRUE)
  #       writeRaster(sd_proj, sd_path, format = "GTiff", overwrite = TRUE,
  #                   options = "COMPRESS=LZW")
  #       
  #       # --- 95% CI rasters (2.5th and 97.5th percentiles) ---
  #       ci_low_path <- paste0(out_base, "/", species_name, "/predictions/stdv/",
  #                             species_name, "_", year, "_ci_low.tif")
  #       ci_high_path <- paste0(out_base, "/", species_name, "/predictions/stdv/",
  #                              species_name, "_", year, "_ci_high.tif")
  #       ci_low_proj  <- calc(proj_stack, fun = function(x) quantile(x, 0.025, na.rm = TRUE))
  #       ci_high_proj <- calc(proj_stack, fun = function(x) quantile(x, 0.975, na.rm = TRUE))
  #       writeRaster(ci_low_proj,  ci_low_path,  format = "GTiff", overwrite = TRUE,
  #                   options = "COMPRESS=LZW")
  #       writeRaster(ci_high_proj, ci_high_path, format = "GTiff", overwrite = TRUE,
  #                   options = "COMPRESS=LZW")
  #     }
  #     cat("    ✓ Saved mean, SD, and 95% CI rasters (", length(projections), "models)\n")
  #   }
  # }
  # 
  # # ---- Maps ----
  # if (savingPlots) {
  #   cat("\nGenerating maps...\n")
  #   kenya_sf <- st_read(kenya_shapefile_path, quiet = TRUE)
  #   kenya_sf <- st_transform(kenya_sf, crs = 4326)
  #   prob_cols <- colorRampPalette(c("#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C",
  #                                   "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"))(100)
  #   for (year in map_years) {
  #     mean_path <- paste0(out_base, "/", species_name, "/predictions/mean/",
  #                         species_name, "_", year, "_mean.tif")
  #     map_path <- paste0(out_base, "/", species_name, "/maps/",
  #                        species_name, "_", year, "_map.pdf")
  #     if (!file.exists(mean_path) || file.exists(map_path)) next
  #     r_df <- as.data.frame(raster(mean_path), xy = TRUE)
  #     colnames(r_df) <- c("x", "y", "prob")
  #     r_df <- r_df[!is.na(r_df$prob), ]
  #     p <- ggplot() +
  #       geom_raster(data = r_df, aes(x = x, y = y, fill = prob)) +
  #       scale_fill_gradientn(colours = prob_cols, limits = c(0,1), name = "Probability") +
  #       geom_sf(data = kenya_sf, fill = NA, colour = "grey40", linewidth = 0.3) +
  #       coord_sf(xlim = c(33.9,41.9), ylim = c(-4.7,5.0)) +
  #       labs(title = paste0(gsub("_", ". ", species_name), " — ", year)) +
  #       theme_minimal()
  #     ggsave(map_path, p, width = 7, height = 6, dpi = map_dpi)
  #     cat("  Map saved:", basename(map_path), "\n")
  #   }
  # }
  
  # =========================================================================
  # Performance Summary (AUC, SI, Boyce only — with 95% CI)
  # =========================================================================
  summary_df <- data.frame(
    species = species_name,
    n_replicates = nberOfReplicates,
    n_successful_ccv = sum(!is.na(AUCs$CCV_AUC)),
    n_successful_scv = sum(!is.na(AUCs$SCV_AUC)),
    mean_CCV_AUC = round(mean(AUCs$CCV_AUC, na.rm = TRUE), 4),
    sd_CCV_AUC = round(sd(AUCs$CCV_AUC, na.rm = TRUE), 4),
    mean_SCV_AUC = round(mean(AUCs$SCV_AUC, na.rm = TRUE), 4),
    sd_SCV_AUC = round(sd(AUCs$SCV_AUC, na.rm = TRUE), 4),
    mean_SI = round(mean(SIs$SI, na.rm = TRUE), 4),
    sd_SI = round(sd(SIs$SI, na.rm = TRUE), 4),
    mean_Boyce = round(mean(Boyce_vals$Boyce, na.rm = TRUE), 4),
    sd_Boyce = round(sd(Boyce_vals$Boyce, na.rm = TRUE), 4)
  )
  
  # >>> CHANGED: 95% CI columns for internal metrics
  summary_df$CI_low_SCV_AUC  <- round(quantile(AUCs$SCV_AUC,  0.025, na.rm = TRUE), 4)
  summary_df$CI_high_SCV_AUC <- round(quantile(AUCs$SCV_AUC,  0.975, na.rm = TRUE), 4)
  summary_df$CI_low_CCV_AUC  <- round(quantile(AUCs$CCV_AUC,  0.025, na.rm = TRUE), 4)
  summary_df$CI_high_CCV_AUC <- round(quantile(AUCs$CCV_AUC,  0.975, na.rm = TRUE), 4)
  summary_df$CI_low_SI       <- round(quantile(SIs$SI,        0.025, na.rm = TRUE), 4)
  summary_df$CI_high_SI      <- round(quantile(SIs$SI,        0.975, na.rm = TRUE), 4)
  summary_df$CI_low_Boyce    <- round(quantile(Boyce_vals$Boyce, 0.025, na.rm = TRUE), 4)
  summary_df$CI_high_Boyce   <- round(quantile(Boyce_vals$Boyce, 0.975, na.rm = TRUE), 4)
  
  write.csv(summary_df, paste0(out_base, "/", species_name, "/performance/performance_summary_internal.csv"),
            row.names = FALSE)
  cat("\nInternal performance summary:\n")
  print(summary_df)
}

# ===========================================================================
# EXTERNAL VALIDATION (20% HOLD-OUT)
# ===========================================================================
cat("\n" , rep("=", 60), "\n")
cat("  EXTERNAL VALIDATION (20% Hold-out Set)\n")
cat(rep("=", 60), "\n")

holdout_rds <- "RDS Data/Ae_aegypti_PA_100_5km_(240926)_test_20_spat_final.rds"

if (!file.exists(holdout_rds)) {
  cat("Hold-out RDS not found at:", holdout_rds, "\n")
  cat("Skipping external validation.\n")
} else {
  cat("Loading hold-out data...\n")
  holdout_list <- readRDS(holdout_rds)
  
  if (length(holdout_list) != length(train_sets)) {
    warning("Hold-out list length differs from training list. Check alignment.")
  }
  
  # ---- Helper functions ----
  clean_holdout_df <- function(df, old_to_new, cov_cols) {
    for (old in names(old_to_new)) {
      if (old %in% colnames(df)) {
        colnames(df)[colnames(df) == old] <- old_to_new[old]
      }
    }
    if ("Longitude" %in% colnames(df) && !("x" %in% colnames(df))) 
      colnames(df)[colnames(df) == "Longitude"] <- "x"
    if ("Latitude" %in% colnames(df) && !("y" %in% colnames(df))) 
      colnames(df)[colnames(df) == "Latitude"] <- "y"
    if (!("response" %in% colnames(df)) && "presence" %in% colnames(df)) 
      colnames(df)[colnames(df) == "presence"] <- "response"
    keep_cols <- c("x", "y", "response", "year", "month", cov_cols)
    keep_cols <- keep_cols[keep_cols %in% colnames(df)]
    df <- df[, keep_cols, drop = FALSE]
    df <- df[complete.cases(df[, cov_cols]), ]
    return(df)
  }
  
  # >>> CHANGED: TSS, Sørensen (unweighted), TP/FP/TN/FN removed
  compute_external_metrics <- function(model, train_df, holdout_df, cov_cols, threshold) {
    holdout_pred <- predict.gbm(model, holdout_df[, cov_cols], 
                                model$gbm.call$best.trees, type = "response")
    
    auc_val <- tryCatch(pROC::roc(holdout_df$response, holdout_pred, quiet = TRUE)$auc, 
                        error = function(e) NA)
    
    # Weighted Sørensen using training prevalence
    TP <- sum(holdout_df$response == 1 & holdout_pred >= threshold)
    FP <- sum(holdout_df$response == 0 & holdout_pred >= threshold)
    FN <- sum(holdout_df$response == 1 & holdout_pred < threshold)
    
    train_P <- sum(train_df$response == 1)
    train_A <- sum(train_df$response == 0)
    train_prev <- train_P / (train_P + train_A)
    x <- (train_P / train_A) * ((1 - train_prev) / train_prev)
    si_weighted <- ifelse((2*TP + (x*FP) + FN) == 0, 0, 
                          (2*TP) / (2*TP + (x*FP) + FN))
    
    return(data.frame(
      External_AUC = auc_val,
      External_Weighted_SI = si_weighted,
      Threshold_Used = threshold,
      N_pres_hold = sum(holdout_df$response == 1),
      N_abs_hold = sum(holdout_df$response == 0)
    ))
  }
  
  valid_reps <- which(!is.na(AUCs$SCV_AUC))
  if (length(valid_reps) == 0) {
    cat("No valid SCV models – skipping external validation.\n")
  } else {
    # >>> CHANGED: external results df now has only AUC, SI, Boyce (+ bookkeeping)
    external_results <- data.frame(
      replicate = 1:nberOfReplicates,
      External_AUC = NA,
      External_Weighted_SI = NA,
      External_Boyce = NA,
      Threshold_Used = NA,
      N_pres_hold = NA,
      N_abs_hold = NA
    )
    
    cat("Computing external metrics for", length(valid_reps), "valid replicates...\n")
    
    for (i in valid_reps) {
      model <- brt_models_scv[[i]]
      if (is.null(model)) next
      
      holdout_df <- holdout_list[[i]]
      if (is.null(holdout_df) || nrow(holdout_df) == 0) {
        cat("  Replicate", i, ": hold-out data empty – skipping\n")
        next
      }
      holdout_df <- clean_holdout_df(holdout_df, old_to_new, cov_cols)
      if (nrow(holdout_df) == 0) {
        cat("  Replicate", i, ": hold-out data has no complete cases – skipping\n")
        next
      }
      
      missing_covs <- setdiff(cov_cols, colnames(holdout_df))
      if (length(missing_covs) > 0) {
        cat("  Replicate", i, ": missing covariates:", paste(missing_covs, collapse=", "), "– skipping\n")
        next
      }
      
      thr <- SIs$threshold[i]
      if (is.na(thr)) {
        # Fallback: use 0.5 if no optimised threshold
        thr <- 0.5
      }
      
      metrics <- compute_external_metrics(
        model = model,
        train_df = as.data.frame(train_sets[[i]]),
        holdout_df = holdout_df,
        cov_cols = cov_cols,
        threshold = thr
      )
      
      external_results[i, c("External_AUC", "External_Weighted_SI", 
                            "Threshold_Used", "N_pres_hold", "N_abs_hold")] <- metrics
    }
    
    # =========================================================================
    # BOYCE INDEX (External / Hold-out data)
    # =========================================================================
    cat("\nComputing Boyce Index for external evaluation...\n")
    boyce_ext_path <- paste0(out_base, "/", species_name, "/performance/Boyce_external.csv")
    
    if (!file.exists(boyce_ext_path)) {
      Boyce_ext_vals <- data.frame(replicate = 1:nberOfReplicates, Boyce = NA)
      
      for (i in valid_reps) {
        model <- brt_models_scv[[i]]
        if (is.null(model)) next
        
        holdout_df <- holdout_list[[i]]
        if (is.null(holdout_df) || nrow(holdout_df) == 0) next
        holdout_df <- clean_holdout_df(holdout_df, old_to_new, cov_cols)
        if (nrow(holdout_df) == 0 || any(!cov_cols %in% colnames(holdout_df))) next
        
        holdout_pred <- predict.gbm(model, holdout_df[, cov_cols], 
                                    model$gbm.call$best.trees, type = "response")
        pres_pred <- holdout_pred[holdout_df$response == 1]
        
        if (length(holdout_pred) < 2 || length(pres_pred) < 2) next
        
        boyce_result <- tryCatch({
          ecospat.boyce(fit = holdout_pred, obs = pres_pred, PEplot = FALSE)
        }, error = function(e) NULL)
        
        if (!is.null(boyce_result) && !is.null(boyce_result$cor) &&
            length(boyce_result$cor) == 1 && is.finite(boyce_result$cor)) {
          Boyce_ext_vals$Boyce[i] <- boyce_result$cor
        } else {
          cat("  Replicate", i, ": Boyce returned NA (skipping)\n")
        }
        
        if (i %% 10 == 0) cat("  Processed", i, "replicates\n")
      }
      write.csv(Boyce_ext_vals, boyce_ext_path, row.names = FALSE)
    } else {
      Boyce_ext_vals <- read.csv(boyce_ext_path)
    }
    
    external_results$External_Boyce <- Boyce_ext_vals$Boyce
    
    # ---- Save external metrics ----
    ext_path <- paste0(out_base, "/", species_name, "/performance/external_validation_metrics.csv")
    write.csv(external_results, ext_path, row.names = FALSE)
    cat("External metrics saved to:", ext_path, "\n")
    
    # ---- External boxplot (AUC, SI, Boyce only) ----
    valid_ext <- external_results[valid_reps, ]
    external_plot_df <- data.frame(
      replicate = valid_ext$replicate,
      AUC = valid_ext$External_AUC,
      SI = valid_ext$External_Weighted_SI,
      Boyce = valid_ext$External_Boyce
    )
    
    external_long <- external_plot_df %>%
      pivot_longer(cols = c(AUC, SI, Boyce), 
                   names_to = "Metric", values_to = "Value")
    external_long$Metric <- factor(external_long$Metric, 
                                   levels = c("AUC", "SI", "Boyce"))
    
    p_ext <- ggplot(external_long, aes(x = Metric, y = Value, fill = Metric)) +
      geom_boxplot(alpha = 0.7) +
      geom_jitter(width = 0.2, size = 0.8, alpha = 0.4) +
      labs(title = paste0(species_name, " – External Performance (20% Hold-out)"),
           y = "Value", x = "") +
      ylim(0, 1) +
      theme_minimal() +
      theme(legend.position = "none")
    
    ggsave(paste0(out_base, "/", species_name, "/performance/external_metrics_boxplot.pdf"),
           p_ext, width = 6, height = 5, dpi = 300)
    cat("  External metrics boxplot saved.\n")
    
    # ---- Combined internal vs external boxplot (AUC, SI, Boyce only) ----
    internal_sub <- internal_metrics[valid_reps, c("replicate", "AUC", "SI", "Boyce")]
    internal_sub$Type <- "Internal"
    
    external_sub <- external_plot_df
    external_sub$Type <- "External"
    
    combined <- rbind(internal_sub, external_sub)
    combined_long <- combined %>%
      pivot_longer(cols = c(AUC, SI, Boyce), 
                   names_to = "Metric", values_to = "Value")
    combined_long$Metric <- factor(combined_long$Metric, 
                                   levels = c("AUC", "SI", "Boyce"))
    
    p_comb <- ggplot(combined_long, aes(x = Metric, y = Value, fill = Type)) +
      geom_boxplot(alpha = 0.7, position = position_dodge(width = 0.8)) +
      labs(title = paste0(species_name, " – Internal vs External Performance"),
           y = "Value", x = "") +
      ylim(0, 1) +
      theme_minimal() +
      theme(legend.position = "top")
    
    ggsave(paste0(out_base, "/", species_name, "/performance/internal_vs_external_boxplots.pdf"),
           p_comb, width = 8, height = 6, dpi = 300)
    cat("  Combined internal vs external boxplots saved.\n")
    
    # =========================================================================
    # EXTERNAL CURVES: ROC & SI – Mean ± 95% CI
    # =========================================================================
    cat("\nComputing external ROC and SI curves across replicates...\n")
    
    thresholds <- seq(0, 1, by = 0.005)
    n_curves_ext <- length(valid_reps)
    
    roc_ext_tpr <- matrix(NA, nrow = length(thresholds), ncol = n_curves_ext)
    roc_ext_fpr <- matrix(NA, nrow = length(thresholds), ncol = n_curves_ext)
    si_ext <- matrix(NA, nrow = length(thresholds), ncol = n_curves_ext)
    
    for (idx in 1:n_curves_ext) {
      i <- valid_reps[idx]
      model <- brt_models_scv[[i]]
      if (is.null(model)) next
      
      holdout_df <- holdout_list[[i]]
      if (is.null(holdout_df) || nrow(holdout_df) == 0) next
      holdout_df <- clean_holdout_df(holdout_df, old_to_new, cov_cols)
      if (nrow(holdout_df) == 0) next
      if (any(!cov_cols %in% colnames(holdout_df))) next
      
      holdout_pred <- predict.gbm(model, holdout_df[, cov_cols], 
                                  model$gbm.call$best.trees, type = "response")
      responses <- holdout_df$response
      
      train_df <- as.data.frame(train_sets[[i]])
      P_train <- sum(train_df$response == 1)
      A_train <- sum(train_df$response == 0)
      prev_train <- P_train / (P_train + A_train)
      x <- (P_train / A_train) * ((1 - prev_train) / prev_train)
      
      for (t in seq_along(thresholds)) {
        th <- thresholds[t]
        tp <- sum(responses == 1 & holdout_pred >= th)
        fp <- sum(responses == 0 & holdout_pred >= th)
        fn <- sum(responses == 1 & holdout_pred < th)
        tn <- sum(responses == 0 & holdout_pred < th)
        roc_ext_tpr[t, idx] <- ifelse((tp + fn) == 0, NA, tp / (tp + fn))
        roc_ext_fpr[t, idx] <- ifelse((fp + tn) == 0, NA, fp / (fp + tn))
        si <- ifelse((2*tp + x*fp + fn) == 0, 0, (2*tp) / (2*tp + x*fp + fn))
        si_ext[t, idx] <- si
      }
      if (idx %% 10 == 0) cat("  Processed", idx, "replicates (external)\n")
    }
    
    mean_roc_ext_tpr <- rowMeans(roc_ext_tpr, na.rm = TRUE)
    mean_roc_ext_fpr <- rowMeans(roc_ext_fpr, na.rm = TRUE)
    ci_roc_ext_tpr_low <- apply(roc_ext_tpr, 1, quantile, 0.025, na.rm = TRUE)
    ci_roc_ext_tpr_high <- apply(roc_ext_tpr, 1, quantile, 0.975, na.rm = TRUE)
    mean_si_ext <- rowMeans(si_ext, na.rm = TRUE)
    ci_si_ext_low <- apply(si_ext, 1, quantile, 0.025, na.rm = TRUE)
    ci_si_ext_high <- apply(si_ext, 1, quantile, 0.975, na.rm = TRUE)
    
    # ---- Plot external curves ----
    roc_ext_df <- data.frame(
      FPR = mean_roc_ext_fpr,
      TPR = mean_roc_ext_tpr,
      CI_low = ci_roc_ext_tpr_low,
      CI_high = ci_roc_ext_tpr_high
    )
    roc_ext_df <- na.omit(roc_ext_df)
    
    p_roc_ext <- ggplot(roc_ext_df, aes(x = FPR, y = TPR)) +
      geom_ribbon(aes(ymin = CI_low, ymax = CI_high), fill = "#D62828", alpha = 0.2) +
      geom_line(color = "#D62828", size = 1.2) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
      labs(x = "1 - Specificity (False Positive Rate)", 
           y = "Sensitivity (True Positive Rate)",
           title = paste0(species_name, " – External ROC Curve (Hold-out)"),
           subtitle = paste0("Mean ± 95% CI (", n_curves_ext, " replicates)")) +
      coord_equal() +
      theme_minimal()
    
    ggsave(paste0(out_base, "/", species_name, "/performance/ROC_external.pdf"),
           p_roc_ext, width = 6, height = 6, dpi = 300)
    
    si_ext_df <- data.frame(
      Threshold = thresholds,
      SI = mean_si_ext,
      CI_low = ci_si_ext_low,
      CI_high = ci_si_ext_high
    )
    si_ext_df <- na.omit(si_ext_df)
    
    p_si_ext <- ggplot(si_ext_df, aes(x = Threshold, y = SI)) +
      geom_ribbon(aes(ymin = CI_low, ymax = CI_high), fill = "#D62828", alpha = 0.2) +
      geom_line(color = "#D62828", size = 1.2) +
      labs(x = "Threshold", 
           y = "Sørensen Index (SI)",
           title = paste0(species_name, " – External SI Curve (Hold-out)"),
           subtitle = paste0("Mean ± 95% CI (", n_curves_ext, " replicates)")) +
      ylim(0, 1) +
      theme_minimal()
    
    ggsave(paste0(out_base, "/", species_name, "/performance/SI_external.pdf"),
           p_si_ext, width = 6, height = 6, dpi = 300)
    
    cat("  External ROC and SI curves saved.\n")
    
    # ---- Combined curves (Internal vs External) ----
    roc_combined_df <- data.frame(
      FPR = c(mean_roc_train_fpr, mean_roc_ext_fpr),
      TPR = c(mean_roc_train_tpr, mean_roc_ext_tpr),
      CI_low = c(ci_roc_train_tpr_low, ci_roc_ext_tpr_low),
      CI_high = c(ci_roc_train_tpr_high, ci_roc_ext_tpr_high),
      Set = rep(c("Internal", "External"), each = length(thresholds))
    )
    roc_combined_df <- na.omit(roc_combined_df)
    
    p_roc_comb <- ggplot(roc_combined_df, aes(x = FPR, y = TPR, colour = Set, fill = Set)) +
      geom_ribbon(aes(ymin = CI_low, ymax = CI_high), alpha = 0.15, colour = NA) +
      geom_line(size = 1.2) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
      scale_colour_manual(values = c("Internal" = "#2E86AB", "External" = "#D62828")) +
      scale_fill_manual(values = c("Internal" = "#2E86AB", "External" = "#D62828")) +
      labs(x = "1 - Specificity (False Positive Rate)", 
           y = "Sensitivity (True Positive Rate)",
           title = paste0(species_name, " – ROC Curves: Internal vs External"),
           subtitle = paste0("Mean ± 95% CI (", n_curves, " internal, ", n_curves_ext, " external replicates)")) +
      coord_equal() +
      theme_minimal() +
      theme(legend.position = c(0.75, 0.25),
            legend.background = element_rect(fill = "white", colour = "grey80"))
    
    ggsave(paste0(out_base, "/", species_name, "/performance/ROC_internal_vs_external.pdf"),
           p_roc_comb, width = 6, height = 6, dpi = 300)
    
    si_combined_df <- data.frame(
      Threshold = rep(thresholds, 2),
      SI = c(mean_si_train, mean_si_ext),
      CI_low = c(ci_si_train_low, ci_si_ext_low),
      CI_high = c(ci_si_train_high, ci_si_ext_high),
      Set = rep(c("Internal", "External"), each = length(thresholds))
    )
    si_combined_df <- na.omit(si_combined_df)
    
    p_si_comb <- ggplot(si_combined_df, aes(x = Threshold, y = SI, colour = Set, fill = Set)) +
      geom_ribbon(aes(ymin = CI_low, ymax = CI_high), alpha = 0.15, colour = NA) +
      geom_line(size = 1.2) +
      scale_colour_manual(values = c("Internal" = "#2E86AB", "External" = "#D62828")) +
      scale_fill_manual(values = c("Internal" = "#2E86AB", "External" = "#D62828")) +
      labs(x = "Threshold", 
           y = "Sørensen Index (SI)",
           title = paste0(species_name, " – SI Curves: Internal vs External"),
           subtitle = paste0("Mean ± 95% CI (", n_curves, " internal, ", n_curves_ext, " external replicates)")) +
      ylim(0, 1) +
      theme_minimal() +
      theme(legend.position = c(0.75, 0.75),
            legend.background = element_rect(fill = "white", colour = "grey80"))
    
    ggsave(paste0(out_base, "/", species_name, "/performance/SI_internal_vs_external.pdf"),
           p_si_comb, width = 6, height = 6, dpi = 300)
    
    cat("  Combined internal vs external curves saved.\n")
    
    # ---- Comparison table (AUC, SI, Boyce only) ----
    comparison_df <- data.frame(
      Metric = c("AUC", "SI", "Boyce"),
      Internal_Mean = c(
        mean(AUCs$SCV_AUC[valid_reps], na.rm = TRUE),
        mean(SIs$SI[valid_reps], na.rm = TRUE),
        mean(Boyce_vals$Boyce[valid_reps], na.rm = TRUE)
      ),
      Internal_SD = c(
        sd(AUCs$SCV_AUC[valid_reps], na.rm = TRUE),
        sd(SIs$SI[valid_reps], na.rm = TRUE),
        sd(Boyce_vals$Boyce[valid_reps], na.rm = TRUE)
      ),
      Internal_CI_low = c(
        quantile(AUCs$SCV_AUC[valid_reps], 0.025, na.rm = TRUE),
        quantile(SIs$SI[valid_reps], 0.025, na.rm = TRUE),
        quantile(Boyce_vals$Boyce[valid_reps], 0.025, na.rm = TRUE)
      ),
      Internal_CI_high = c(
        quantile(AUCs$SCV_AUC[valid_reps], 0.975, na.rm = TRUE),
        quantile(SIs$SI[valid_reps], 0.975, na.rm = TRUE),
        quantile(Boyce_vals$Boyce[valid_reps], 0.975, na.rm = TRUE)
      ),
      External_Mean = c(
        mean(valid_ext$External_AUC, na.rm = TRUE),
        mean(valid_ext$External_Weighted_SI, na.rm = TRUE),
        mean(valid_ext$External_Boyce, na.rm = TRUE)
      ),
      External_SD = c(
        sd(valid_ext$External_AUC, na.rm = TRUE),
        sd(valid_ext$External_Weighted_SI, na.rm = TRUE),
        sd(valid_ext$External_Boyce, na.rm = TRUE)
      ),
      External_CI_low = c(
        quantile(valid_ext$External_AUC, 0.025, na.rm = TRUE),
        quantile(valid_ext$External_Weighted_SI, 0.025, na.rm = TRUE),
        quantile(valid_ext$External_Boyce, 0.025, na.rm = TRUE)
      ),
      External_CI_high = c(
        quantile(valid_ext$External_AUC, 0.975, na.rm = TRUE),
        quantile(valid_ext$External_Weighted_SI, 0.975, na.rm = TRUE),
        quantile(valid_ext$External_Boyce, 0.975, na.rm = TRUE)
      ),
      Drop = c(
        mean(valid_ext$External_AUC, na.rm = TRUE) - mean(AUCs$SCV_AUC[valid_reps], na.rm = TRUE),
        mean(valid_ext$External_Weighted_SI, na.rm = TRUE) - mean(SIs$SI[valid_reps], na.rm = TRUE),
        mean(valid_ext$External_Boyce, na.rm = TRUE) - mean(Boyce_vals$Boyce[valid_reps], na.rm = TRUE)
      )
    )
    
    print(comparison_df)
    comp_path <- paste0(out_base, "/", species_name, "/performance/internal_vs_external_comparison.csv")
    write.csv(comparison_df, comp_path, row.names = FALSE)
    cat("\nComparison saved to:", comp_path, "\n")
    
    # ---- Update summary with external means and 95% CI ----
    summary_df$mean_External_AUC <- round(mean(valid_ext$External_AUC, na.rm = TRUE), 4)
    summary_df$sd_External_AUC <- round(sd(valid_ext$External_AUC, na.rm = TRUE), 4)
    summary_df$CI_low_External_AUC  <- round(quantile(valid_ext$External_AUC, 0.025, na.rm = TRUE), 4)
    summary_df$CI_high_External_AUC <- round(quantile(valid_ext$External_AUC, 0.975, na.rm = TRUE), 4)
    
    summary_df$mean_External_SI <- round(mean(valid_ext$External_Weighted_SI, na.rm = TRUE), 4)
    summary_df$sd_External_SI <- round(sd(valid_ext$External_Weighted_SI, na.rm = TRUE), 4)
    summary_df$CI_low_External_SI  <- round(quantile(valid_ext$External_Weighted_SI, 0.025, na.rm = TRUE), 4)
    summary_df$CI_high_External_SI <- round(quantile(valid_ext$External_Weighted_SI, 0.975, na.rm = TRUE), 4)
    
    summary_df$mean_External_Boyce <- round(mean(valid_ext$External_Boyce, na.rm = TRUE), 4)
    summary_df$sd_External_Boyce <- round(sd(valid_ext$External_Boyce, na.rm = TRUE), 4)
    summary_df$CI_low_External_Boyce  <- round(quantile(valid_ext$External_Boyce, 0.025, na.rm = TRUE), 4)
    summary_df$CI_high_External_Boyce <- round(quantile(valid_ext$External_Boyce, 0.975, na.rm = TRUE), 4)
    
    write.csv(summary_df, paste0(out_base, "/", species_name, "/performance/performance_summary_full.csv"),
              row.names = FALSE)
    cat("\nFull performance summary (internal + external):\n")
    print(summary_df)
  }
}

cat("\n╔════════════════════════════════════════════════════════════╗\n")
cat("  PIPELINE COMPLETE (no standardisation)\n")
cat("╚════════════════════════════════════════════════════════════╝\n")



train_new <- readRDS("BRT_results/Ae_aegypti/models/BRT_models_SCV.rds")
train_old <- readRDS("BRT_results/Ae_aegypti_5km_100models_with_buffer_bias_10covs_SCV_final_final_lr0.01_v2_(240926)/models/BRT_models_SCV.rds")
