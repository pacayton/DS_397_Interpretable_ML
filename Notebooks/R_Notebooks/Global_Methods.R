## Preamble

# install.packages(c("iml", "randomForest","rpart" ,"nnet", "tidyverse", "GGally", "Metrics", "e1071", "glmnet", "partykit", 'latex2exp', "patchwork"))


library(iml)
library(randomForest)
library(nnet)
library(tidyverse)
library(GGally)
library(Metrics)
library(rpart)
library(glmnet)
library(patchwork)
library(partykit)
library(mgcv)

## Penguins Dataset

penguins <- read.csv("https://github.com/christophM/interpretable-ml-book/raw/refs/heads/master/data/penguins.csv")

set.seed(2)
penguins = na.omit(penguins)
penguins$sex = as.factor(penguins$sex)

penguins_orig = penguins
penguins_col_selection = c('species', 'bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'sex', 'body_mass_g') 

penguins = penguins[penguins_col_selection] 

penguin_train_index <- sample(1:nrow(penguins), 2/3 * nrow(penguins))
penguins_train <- penguins[penguin_train_index, ]
penguins_test <- penguins[-penguin_train_index, ]

# Make sure P(female) is predicted
penguins$sex <- relevel(penguins$sex, ref = "male")
pengu_rf = randomForest(sex ~ ., data = penguins_train)
pengu_tree = rpart(sex ~ ., data = penguins_train)

# Split the data by species
species_models <- lapply(split(penguins_train, penguins_train$species), function(data) {
  data$species = NULL
  # ensuring the right levels
  data$sex <- relevel(data$sex, ref = "male")
  glm(sex ~ ., data = data, trace = FALSE, family = binomial(link = "logit"))
})

pengu_logreg <- list(models = species_models)
class(pengu_logreg) <- "pengu_logreg"

predict.pengu_logreg <- function(object, newdata, ...) {
  predictions <- numeric(nrow(newdata))
  
  for (species in names(object$models)) {
    # Filter test data for this species
    species_data <- newdata[newdata$species == species, ]
    if (nrow(species_data) > 0) {
      # Predict using the appropriate model
      model <- object$models[[species]]
      species_predictions <- predict(model, newdata = species_data, type = "response", ...)
      
      # Create temporary data frame with predictions
      pred_df <- data.frame(
        row_id = seq_len(nrow(newdata))[newdata$species == species],
        pred = species_predictions
      )
      # Update predictions using merge matching
      predictions[pred_df$row_id] <- pred_df$pred
    }
  }
  
  return(predictions)
}

## Bike Dataset

set.seed(42)
bike <- read.csv("https://github.com/christophM/interpretable-ml-book/raw/refs/heads/master/data/bike.csv")

bike = na.omit(bike)
bike$weather <- as.factor(bike$weather)

bike_features =  c('season','holiday', 'workday', 'weather', 'temp', 'hum', 'windspeed',  "cnt_2d_bfr")

bike = bike[c(bike_features, "cnt")]

bike_train_index <- sample(1:nrow(bike), 2/3 * nrow(bike))
bike_train <- bike[bike_train_index, ]
bike_test <- bike[-bike_train_index, ]

bike_rf = randomForest(cnt ~ ., data = bike_train)
bike_tree = rpart(cnt ~ ., data = bike_train)
bike_svm = e1071::svm(cnt ~ ., data = bike_train)
bike_lm = lm(cnt ~ ., data = bike_train, x = TRUE)

## Partial Dependence Plots

### Example: Bike Data using Random Forest 

pred.bike = Predictor$new(bike_rf, data = bike_test)

pdp = FeatureEffect$new(pred.bike, "temp", method = "pdp")

p1 = pdp$plot() +
  scale_x_continuous('Temperature', limits = c(0, NA)) +
  scale_y_continuous('Predicted number of bikes', limits = c(0, 5500))
print(p1)

pdp$set.feature("hum")
p2 = pdp$plot() +
  scale_x_continuous('Humidity', limits = c(0, NA)) +
  scale_y_continuous('', limits = c(0, 5500))
print(p2)

pdp$set.feature("windspeed")
p3 = pdp$plot() +
  scale_x_continuous('Wind speed', limits = c(0, NA)) +
  scale_y_continuous('', limits = c(0, 5500))

pdp = FeatureEffect$new(pred.bike, "season", method = "pdp")
ggplot(pdp$results) +
  geom_col(aes(x = season, y = .value), width = 0.3) +
  scale_x_discrete('Season') +
  scale_y_continuous('', limits = c(0, 5500))

pdp = FeatureEffect$new(pred.bike, "cnt_2d_bfr", method = "pdp")
p4 = pdp$plot() +
  scale_x_continuous('Count 2 Days Before', limits = c(0, NA)) +
  scale_y_continuous('Predicted number of bikes', limits = c(0, 6000))
print(p4)

pdp = FeatureEffect$new(pred.bike, "workday", method = "pdp")
ggplot(pdp$results) +
  geom_col(aes(x = workday, y = .value), width = 0.3) +
  scale_x_discrete('Workday') +
  scale_y_continuous('', limits = c(0, 5000))

pdp = FeatureEffect$new(pred.bike, "weather", method = "pdp")
ggplot(pdp$results) +
  geom_col(aes(x = weather, y = .value), width = 0.3) +
  scale_x_discrete('Weather') +
  scale_y_continuous('', limits = c(0, 5000))

## get PDP Values
## install.packages("pdp")
pdp_values_cnt_2d_bfr <- pdp::partial(bike_rf, pred.var = "cnt_2d_bfr")

## install.packages("DALEX")
library("DALEX")
## Explainer Object
explainer_bike_rf <- explain(bike_rf, data = bike_test[, -9], y = bike_test[,9])
### PDP Profiles (Values)
pdp_profile_c2b <- model_profile(explainer_bike_rf, variables = "cnt_2d_bfr")

### Penguin Dataset

pred_rf <- Predictor$new(pengu_rf, data = penguins, class = "female")
# Function to get PDP data
get_pdp_data <- function(predictor, feature) {
  pdp <- FeatureEffect$new(predictor, feature, method = "pdp")
  data <- as.data.frame(pdp$results)
  data$feature <- feature
  data$model <- class(predictor$model)[1]
  names(data)[names(data) == ".value"] <- "probability"
  names(data)[names(data) == feature] <- "value"
  return(data)
}

# c('species', 'bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'sex', 'body_mass_g')

pdp_bill_depth <- get_pdp_data(pred_rf, "bill_depth_mm")

ggplot(pdp_bill_depth, aes(x = value, y = probability)) +
  geom_line() +
  scale_y_continuous("P(female)", limits = c(0,1)) +
  scale_x_continuous("Bill Depth in Millimeters")

## PDP for 2 Variables

pdp_2var <- FeatureEffect$new(pred.bike, c("windspeed", "temp"), method = "pdp")

pdp_2var$plot() + 
  geom_point(data = bike_test) 

## Accumulated Local Effects

### Bike Data using Random Forests

pred.bike <- Predictor$new(bike_rf, data = bike_test, y = "cnt")
limits <- c(-800, 250)

#### ALE for Windspeed

ale_wspd <- FeatureEffect$new(pred.bike, "windspeed", method = "ale")
ale_wspd$plot() +
  scale_x_continuous("Wind Speed") +
  scale_y_continuous("ALE of Bike Rentals", limits = limits)

#### ALE for Weather

# alecat1 <- FeatureEffect$new(pred.bike , "weather", method = "ale")
# ggplot(alecat1$results) + 
#   geom_col(aes(x = weather, y = .value), width = 0.3) +
#   geom_hline(yintercept = 0) +
#   scale_x_discrete("Weather") +
#   scale_y_continuous("ALE", limits = limits)
# 
# alecat1$results

## install.packages("yaImpute")

ale_2var <- FeatureEffect$new(pred.bike, c("windspeed","temp"), method = "ale")
ale_2var$plot() +
  geom_point(data = bike_test)
  
## Feature Interaction

### Penguin Data with Random Forest

pred.peng <- Predictor$new(pengu_rf, data = penguins, class= "female")
ia1 = Interaction$new(pred.peng, grid.size = 30)
plot(ia1)

ia2 = Interaction$new(pred.peng, grid.size = 30, feature = "body_mass_g")
plot(ia2)

## Bike Data using Random Forest

pred_bike <- Predictor$new(bike_rf, data = bike)
ia1_bike = Interaction$new(pred.bike, grid.size = 30)
plot(ia1_bike)

ia2_c2b <- Interaction$new(pred.bike, grid.size = 30, feature = "cnt_2d_bfr")
plot(ia2_c2b)

ale_2var <- FeatureEffect$new(pred.bike, c("cnt_2d_bfr", "hum"), method = "ale")

ale_2var$results

ale_2var$plot() + 
  geom_point(data = bike_test)

## Functional Decomposition

## Permutation Feature Importance

### Bike Dataset using SVM

pred.bike = Predictor$new(bike_svm, data = bike_test[,-9], y = bike_test$cnt)

imp.bike <- FeatureImp$new(pred.bike , loss = "mae")
(imp.bike_dat <- imp.bike$results)

plot(imp.bike)

### Penguins Data 

pred.pgns <- Predictor$new(pengu_rf, data = penguins_test %>% select(-sex), y = (penguins_test$sex == "female"), class = "female" )

imp.pgns <- FeatureImp$new(pred.pgns, loss = "ce", compare = "difference")

(imp.pgns_dat <- imp.pgns$results)

plot(imp.pgns)

#### Group-Based PFI

penguins_by_species <- penguins %>%
  mutate(subset = "all") %>%
  bind_rows(penguins %>% mutate(subset = species))

cor_data <- penguins_by_species %>%
  group_by(subset) %>%
  dplyr::summarize(cor = cor(flipper_length_mm, body_mass_g, use = "complete.obs")) %>%
  mutate(cor_label = paste("r =", round(cor, 2)))

ggplot(penguins_by_species) + 
  geom_point(aes(x = flipper_length_mm, y = body_mass_g, color = species)) + 
  facet_wrap(~subset, scales = "free") + 
  geom_label(data = cor_data, aes(x = -Inf, y = Inf, label = cor_label), 
             hjust = -0.1, vjust = 1.5, inherit.aes = FALSE) + 
  theme_minimal() +
  scale_x_continuous("Flipper length in mm") +
  scale_y_continuous("Body mass in grams") 

### 
dat = lapply(c("Adelie", "Chinstrap", "Gentoo"), 
             function(species_) {
               penguins_sub = penguins_test %>% 
                 dplyr::filter(species == species_
                               )
               pred.pgns = Predictor$new(pengu_rf, y = penguins_sub$sex == "female", data = penguins_sub%>% select(-sex), class="female")
  dat = FeatureImp$new(pred.pgns,  loss = "ce", compare="ratio")$results
  dat$species = species_
  dat
})

dat = do.call(rbind, dat)

ggplot(dat) + 
  geom_point(aes(x = importance, y = feature, shape = species, color = species), 
             size = 3, position = position_jitter(width = 0, height = 0.2)) +
  scale_x_continuous("Feature importance (ratio of ce)")

## LOFO Importance
## Using the Bike Dataset
## Predictions based on Random Forest Model

test_prediction <- predict(bike_rf, newdata = bike_test)
test_mae <- mean(abs(bike_test$cnt - test_prediction))

feature_importances <- data.frame(feature = setdiff(names(bike_train), 'cnt'), importance = 0)

for (feature in feature_importances$feature) {
  new_train_data <- bike_train %>% select(-!!sym(feature))
  new_mod = randomForest(cnt ~ ., data = new_train_data)
  new_test_prediction <- predict(new_mod, newdata = bike_test %>% select(-!!sym(feature)))
  new_test_mae <- mean(abs(bike_test$cnt - new_test_prediction))
  feature_importances$importance[feature_importances$feature == feature] <- new_test_mae - test_mae 
  
}

# Order the features by importance
feature_importances <- feature_importances %>%
  arrange(desc(importance))

# Visualize the feature importances
ggplot(feature_importances, aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(x = "Feature", y = "LOFO Importance (MAE difference)")

# Using the Penguins Dataset

test_prediction <- predict(pengu_rf, newdata = penguins_test)
test_accuracy <- mean(penguins_test$sex == test_prediction)

feature_importances <- data.frame(feature = setdiff(names(penguins), 'sex'), importance = 0)

for (feature in feature_importances$feature) {
  new_train_data <- penguins_train %>% select(-!!sym(feature))
  new_rf_model <- randomForest(sex ~ ., data = new_train_data)
  new_test_prediction <- predict(new_rf_model, newdata = penguins_test %>% select(-!!sym(feature)))
  new_test_accuracy <- mean(penguins_test$sex == new_test_prediction)
  feature_importances$importance[feature_importances$feature == feature] <- test_accuracy - new_test_accuracy
}

# Order the features by importance
feature_importances <- feature_importances %>%
  arrange(desc(importance))

# Visualize the feature importances
ggplot(feature_importances, aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(x = "", y = "LOFO Importance (accuracy difference)") 

## Surrogate Models

### Bike Dataset
## with Support Vector Machine predictions
bike_train$holiday <- as.factor(bike_train$holiday)
bike_train$season <- as.factor(bike_train$season)
bike_train$workday <- as.factor(bike_train$workday)
bike_test$holiday <- as.factor(bike_test$holiday)
bike_test$season <- as.factor(bike_test$season)
bike_test$workday <- as.factor(bike_test$workday)

pred.bike = Predictor$new(bike_svm, data = bike_train)
tree = TreeSurrogate$new(pred.bike) 
pred.tree  = predict(tree, bike_test)
pred.svm = predict(bike_svm, bike_test)
rsq = round(cor(pred.tree, pred.svm)^2, 2)

plot(tree)

### GAM as Surrogate

gam_test <- (cbind(pred.svm, bike_test))[,-10]

gam_surr <- gam(pred.svm ~ season + holiday + workday + weather + s(temp) + s(hum) + s(windspeed) + s(cnt_2d_bfr), data = gam_test)

pred.gam <- predict(gam_surr)

rsq_gam = round(cor(pred.gam, pred.svm)^2, 2)

plot(gam_surr)

### Penguins Dataset with Random Forest as the Prediction Model
penguins_newtrain <- penguins_train[,-1]
penguins_newtest <- penguins_test[,-1]
pengu_newrf = randomForest(sex ~ ., data = penguins_newtrain)

pred.penguins = Predictor$new(pengu_newrf, data = penguins_newtrain, type = "prob")
tree.penguins = TreeSurrogate$new(pred.penguins, maxdepth = 2) 
pred.tree.penguins = predict(tree.penguins, penguins_newtest, type="prob")
pred.penguins= predict(pengu_newrf, penguins_newtest, type="prob")
rsq = round(cor(pred.tree.penguins[,"female"], pred.penguins[,"female"])^2, 2)

plot(tree.penguins)

## Prototypes and Criticisms

set.seed(1)
dat1 = data.frame(x1 = rnorm(20, mean = 4, sd = 0.3), x2 = rnorm(20, mean = 1, sd = 0.3))
dat2 = data.frame(x1 = rnorm(30, mean = 2, sd = 0.2), x2 = rnorm(30, mean = 2, sd = 0.2))
dat3 = data.frame(x1 = rnorm(40, mean = 3, sd = 0.2), x2 = rnorm(40, mean = 3))
dat4 = data.frame(x1 = rnorm(7, mean = 4, sd = 0.1), x2 = rnorm(7, mean = 2.5, sd = 0.1))

dat = rbind(dat1, dat2, dat3, dat4)
dat$type = "data"
dat$type[c(7, 23, 77)] = "prototype"
dat$type[c(81,95)] = "criticism"

ggplot(dat, aes(x = x1, y = x2)) + geom_point(alpha = 0.7) +
  geom_point(data = filter(dat, type!='data'), aes(shape = type), size = 6, alpha = 1, color = "blue") +
  scale_shape_manual(breaks = c("prototype", "criticism"), values = c(18, 19)) +
  scale_x_continuous(latex2exp::TeX(r'($X_1$)')) +
  scale_y_continuous(latex2exp::TeX(r'($X_2$)'))

## MMD2

set.seed(42)
n = 40
# create dataset from three gaussians in 2d
dt1 = data.frame(x1 = rnorm(n, mean = 1, sd = 0.1), x2 = rnorm(n, mean = 1, sd = 0.3))
dt2 = data.frame(x1 = rnorm(n, mean = 4, sd = 0.3), x2 = rnorm(n, mean = 2, sd = 0.3))
dt3 = data.frame(x1 = rnorm(n, mean = 3, sd = 0.5), x2 = rnorm(n, mean = 3, sd = 0.3))
dt4 = data.frame(x1 = rnorm(n, mean = 2.6, sd = 0.1), x2 = rnorm(n, mean = 1.7, sd = 0.1))
dt = rbind(dt1, dt2, dt3, dt4)


radial = function(x1, x2, sigma = 1) {
  dist = sum((x1 - x2)^2)
  exp(-dist/(2*sigma^2))
}


cross.kernel = function(d1, d2) {
  kk = c()
  for (i in 1:nrow(d1)) {
    for (j in 1:nrow(d2)) {
      res = radial(d1[i,], d2[j,])
      kk = c(kk, res)
    }
  }
  mean(kk)
}

mmd2 = function(d1, d2) {
  cross.kernel(d1, d1) - 2 * cross.kernel(d1, d2) + cross.kernel(d2,d2)
}

# create 3 variants of prototypes
pt1 = rbind(dt1[c(1,2),], dt4[1,])
pt2 = rbind(dt1[1,], dt2[3,], dt3[19,])
pt3 = rbind(dt2[3,], dt3[19,])

# create plot with all data and density estimation
p = ggplot(dt, aes(x = x1, y = x2)) +
  stat_density_2d(geom = "tile", aes(fill = ..density..), contour = FALSE, alpha = 0.9) +
  geom_point() +
  scale_fill_gradient2(low = "white", high = "blue", guide = "none") +
  scale_x_continuous(latex2exp::TeX(r'($X_1$)'), limits = c(0, NA)) +
  scale_y_continuous(latex2exp::TeX(r'($X_2$)'), limits = c(0, NA))

# create plot for each prototype
p1 = p + geom_point(data = pt1, color = "red", size = 4) + geom_density_2d(data = pt1, color = "red") +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt1)))

p2 = p + geom_point(data = pt2, color = "red", size = 4) +
  geom_density_2d(data = pt2, color = "red") +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt2)))

p3 = p + geom_point(data = pt3, color = "red", size = 4) +
  geom_density_2d(data = pt3, color = "red") +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt3)))
# TODO: Add custom legend for prototypes

# overlay mmd measure for each plot

(p | p1) / (p2 | p3)


witness = function(x, dist1, dist2, sigma = 1) {
  k1 = apply(dist1, 1, function(z) radial(x, z, sigma = sigma))
  k2 = apply(dist2, 1, function(z) radial(x, z, sigma = sigma))
  mean(k1) - mean(k2)
}

w.points.indices = c(125, 2, 60, 19, 100)
wit.points = dt[w.points.indices,]
wit.points$witness = apply(wit.points, 1, function(x) round(witness(x[c("x1", "x2")], dt, pt2, sigma = 1), 3))

p + geom_point(data = pt2, color = "red") +
  geom_density_2d(data = pt2, color = "red") +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt2))) +
  geom_label(data = wit.points, aes(label = witness), alpha = 0.9, vjust = "top") +
  geom_point(data = wit.points, color = "black", shape = 17, size = 4) +
  scale_x_continuous(latex2exp::TeX(r'($X_1$)'), limits = c(0, NA)) +
  scale_y_continuous(latex2exp::TeX(r'($X_2$)'), limits = c(0, NA))