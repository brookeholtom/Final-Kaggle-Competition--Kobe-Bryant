library(patchwork)
library(timetk)
library(tidyverse)
library(ggplot2)
library(vroom)
library(tidymodels)
library(embed)
library(ranger)
library(discrim)
library(naivebayes)
library(kknn)
library(themis)
library(forecast)
library(modeltime)

kobe <- vroom("data.csv")

kobe$time_remaining = (kobe$minutes_remaining*60)+kobe$seconds_remaining

kobe$matchup = ifelse(str_detect(kobe$matchup, 'vs.'), 'Home', 'Away')

kobe['season'] <- substr(str_split_fixed(kobe$season, '-',2)[,2],2,2)

kobe <- kobe %>%
  select(-c( 'team_id', 'team_name', 'shot_zone_range', 'lon', 'lat', 
            'seconds_remaining', 'minutes_remaining', 'game_event_id', 
            'game_id', 'loc_x', 'loc_y'))

# Train
train <- kobe %>%
  filter(!is.na(shot_made_flag))
# Test 
test <- kobe %>% 
  filter(is.na(shot_made_flag))


#Random Forest
my_recipe <- recipe(shot_made_flag ~ ., data = train) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  update_role(shot_id, new_role = "ID") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_naomit()

my_mod_forest <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Create a workflow with model & recipe
workflow_forest <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_forest)

## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1, ncol(train) - 1)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_forest <- workflow_forest %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse))

## Find best tuning parameters
bestTune <- CV_results_forest %>%
  select_best("rmse")

collect_metrics(CV_results_forest) 

## Finalize workflow and predict

final_wf_forest <- workflow_forest %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

predictions_forest <- final_wf_forest %>%
  predict(test)

predictions_forest <- predictions_forest %>%
  bind_cols(., test) %>%
  select(shot_id, .pred) %>%
  rename(shot_made_flag = .pred)

vroom_write(x= predictions_forest, file="predictions_forest_final.csv", delim=",")



#Facebook Prophet Model 
train2 <- vroom("data.csv") %>%
  filter(!is.na(shot_made_flag))  %>%
  select(-c('team_id', 'team_name', 'shot_zone_range', 'lon', 'lat', 
            'seconds_remaining', 'minutes_remaining', 'game_event_id', 
            'game_id','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y'))


test2 <- vroom("data.csv") %>%
  filter(is.na(shot_made_flag))  %>%
  select(-c('team_id', 'team_name', 'shot_zone_range', 'lon', 'lat', 
            'seconds_remaining', 'minutes_remaining', 'game_event_id', 
            'game_id','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y'))

cv_split <- time_series_split(train2, assess='3 months', cumulative = TRUE)

prophet_model <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(shot_made_flag ~ game_date, data = training(cv_split))

cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))

cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

prophet_fullfit <- cv_results %>%
  modeltime_refit(data = train2)

prophet_preds <- prophet_fullfit %>%
  modeltime_forecast(new_data = test2) %>%
  rename(game_date=.index, shot_made_flag=.value) %>%
  select(game_date, shot_made_flag) %>%
  full_join(., y = test2, by=("game_date")) %>%
  select(game_date, shot_made_flag.x) %>%
  rename(shot_made_flag = shot_made_flag.x) 

join <- left_join(test2, prophet_preds, by=("game_date")) %>%
  select(shot_id, shot_made_flag.y) %>%
  rename(shot_made_flag = shot_made_flag.y) %>%
  group_by(shot_id) %>%
  summarise(shot_made_flag = mean(shot_made_flag))
  

vroom_write(x= join, file="predictions_prophet.csv", delim=",")


