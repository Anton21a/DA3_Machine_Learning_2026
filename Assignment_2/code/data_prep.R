library(haven)
library(glmnet)
library(purrr)
library(margins)
library(skimr)
library(kableExtra)
library(Hmisc)
library(cowplot)
library(gmodels) 
library(lspline)
library(sandwich)
library(modelsummary)
library(rattle)
library(caret)
library(pROC)
library(ranger)
library(rpart)
library(partykit)
library(rpart.plot)
library(readr)
library(dplyr)
library(tidyr)

rm(list=ls())

setwd("C:/Users/user/Desktop/MA2y/Data_Analysis_3/DA3_Machine_Learning_2026/Assignment_2/data")

data <- read_csv("cs_bisnode_panel.csv")

# drop variables witр many NAs
data <- data %>%
  select(-c(COGS, finished_prod, net_dom_sales, net_exp_sales, wages)) %>%
  filter(year !=2016)


data <- data %>%
  complete(year, comp_id)


data  <- data %>%
  mutate(status_alive = sales > 0 & !is.na(sales) %>%
           as.numeric(.))

data  <- data %>%
  filter(year == 2012 | year == 2014)

data <- data %>%
  mutate(sales = ifelse(sales < 0, 1, sales))

data <- data %>%
  mutate(age = (year - founded_year) %>%
           ifelse(. < 0, 0, .),
         new = as.numeric(age <= 1) %>% #  (age could be 0,1 )
           ifelse(balsheet_notfullyear == 1, 1, .))

data  <- data %>%
  filter(status_alive == 1) %>%
  filter(!(sales > 10000000)) %>%
  filter(!(sales < 1000))

data <- data %>%
  mutate(ind2_cat = ind2 %>%
           ifelse(. > 56, 60, .)  %>%
           ifelse(. < 26, 20, .) %>%
           ifelse(. < 55 & . > 35, 40, .) %>%
           ifelse(. == 31, 30, .) %>%
           ifelse(is.na(.), 99, .)
  )

table(data$ind2_cat)

data <- data %>%
  mutate(age2 = age^2,
         foreign_management = as.numeric(foreign >= 0.5),
         gender_m = factor(gender, levels = c("female", "male", "mix")),
         m_region_loc = factor(region_m, levels = c("Central", "East", "West")))

data <-data  %>%
  mutate(flag_asset_problem=ifelse(intang_assets<0 | curr_assets<0 | fixed_assets<0,1,0  ))
table(data$flag_asset_problem)

data <- data %>%
  mutate(intang_assets = ifelse(intang_assets < 0, 0, intang_assets),
         curr_assets = ifelse(curr_assets < 0, 0, curr_assets),
         fixed_assets = ifelse(fixed_assets < 0, 0, fixed_assets))

data <- data %>%
  mutate(total_assets_bs = intang_assets + curr_assets + fixed_assets)

sum(is.na(data$sales))

should_not_be_negative <- c(
  "sales",
  "curr_assets",
  "curr_liab",
  "extra_inc",
  "fixed_assets",
  "intang_assets",
  "inventories",
  "liq_assets",
  "material_exp",
  "personnel_exp",
  "subscribed_cap",
  "tang_assets",
  "labor_avg",
  "total_assets_bs",
  "amort",
  "extra_exp"
)

#delete negative values that shouldn't be negative
data <- data %>%
  filter(if_all(all_of(should_not_be_negative), ~ . >= 0 | is.na(.)))

for (var in should_not_be_negative) {
  new_var <- paste0("ln_", var)
  data[[new_var]] <- log1p(data[[var]])  
}

data <- data %>%
  select(-all_of(setdiff(should_not_be_negative, c("sales", "total_assets_bs"))))

ln_should_not_be_negative <- c(
  "ln_sales",
  "ln_curr_assets",
  "ln_curr_liab",
  "ln_extra_inc",
  "ln_fixed_assets",
  "ln_intang_assets",
  "ln_inventories",
  "ln_liq_assets",
  "ln_material_exp",
  "ln_personnel_exp",
  "ln_subscribed_cap",
  "ln_tang_assets",
  "ln_labor_avg",
  "ln_total_assets_bs",
  "ln_amort",
  "ln_extra_exp"
)

can_be_negative <- c(
  "inc_bef_tax",
  "extra_profit_loss",
  "profit_loss_year",
  "share_eq"
)



subset2012 <- subset(data, data$year == 2012)
subset2014 <- subset(data, data$year == 2014)

subset2014 <- left_join(subset2014, subset2012 %>%
                          select(comp_id, total_assets_bs_2012 = total_assets_bs),
                        by = "comp_id")

subset2012$inc_bef_tax <- subset2012$inc_bef_tax / subset2012$total_assets_bs
subset2012$extra_profit_loss <- subset2012$extra_profit_loss / subset2012$total_assets_bs
subset2012$profit_loss_year <- subset2012$profit_loss_year / subset2012$total_assets_bs
subset2012$share_eq <- subset2012$share_eq / subset2012$total_assets_bs

subset2014$inc_bef_tax <- subset2014$inc_bef_tax / subset2014$total_assets_bs_2012
subset2014$extra_profit_loss <- subset2014$extra_profit_loss / subset2014$total_assets_bs_2012
subset2014$profit_loss_year <- subset2014$profit_loss_year / subset2014$total_assets_bs_2012
subset2014$share_eq <- subset2014$share_eq / subset2014$total_assets_bs_2012

subset2014$total_assets_bs <- NULL
subset2014$total_assets_bs_2012 <- NULL
subset2012$total_assets_bs <- NULL

rename_v <- function(data, vars, suffix = "_2012") {
  data %>%
    rename_with(.cols = all_of(vars),
                .fn = ~ paste0(., suffix))
}

total_list2012 <- c(ln_should_not_be_negative, can_be_negative, "ceo_count",
                    "status_alive", "foreign_management", "sales")

subset2012 <- rename_v(subset2012, total_list2012)

subset2012 <- subset2012 %>%
  select(comp_id, ends_with("_2012"))

data_total <- inner_join(subset2014, subset2012, by = "comp_id")

listi <- c('begin', 'end', 'D', 'balsheet_flag', 'balsheet_length',
           'balsheet_notfullyear', 'exit_year', 'inoffice_days',
           'nace_main', 'ind2', 'ind', 'region_m', 'exit_date')

dropping <- function(data, vars) {
  data[, !(names(data) %in% vars)]
}

data_total <- dropping(data_total, listi)

 

data_total$ln_labor_avg <- NULL
data_total$ln_labor_avg_2012 <- NULL
data_total$birth_year <- NULL
data_total$ceo_count_2012 <- NULL
data_total$foreign_management_2012 <- NULL

data_total <- data_total %>%
  filter(!is.na(female))

na_counts <- sapply(data_total, function(x) sum(is.na(x)))
na_counts
na_table <- data.frame(
  variable = names(na_counts),
  na_count = na_counts
)
na_table <- na_table[order(-na_table$na_count), ]

data_total <- data_total %>%
  drop_na()
str(data_total)

growth_vars <- c(
  "ln_sales",
  "ln_curr_assets",
  "ln_curr_liab",
  "ln_extra_inc",
  "ln_fixed_assets",
  "ln_intang_assets",
  "ln_inventories",
  "ln_liq_assets",
  "ln_material_exp",
  "ln_personnel_exp",
  "ln_subscribed_cap",
  "ln_tang_assets",
  "ln_total_assets_bs",
  "ln_amort",
  "ln_extra_exp")


#FINDING log GROWTH RATE
for (var in growth_vars) {
  var_2012 <- paste0(var, "_2012")            
  new_var <- paste0("growth_", var)           
  
  if (var %in% names(data_total) && var_2012 %in% names(data_total)) {
    data_total[[new_var]] <- (data_total[[var]] - data_total[[var_2012]]) }
}

diff_vars <- c(
  "extra_profit_loss",
  "inc_bef_tax",
  "profit_loss_year",
  "share_eq")

for (var in diff_vars) {
  var_2012 <- paste0(var, "_2012")
  new_var <- paste0("diff_", var)
  
  if (all(c(var, var_2012) %in% names(data_total))) {
    data_total[[new_var]] <- data_total[[var]] - data_total[[var_2012]]
  }
}

vars_to_delete <- c(
  growth_vars,
  paste0(growth_vars, "_2012"),
  diff_vars,
  paste0(diff_vars, "_2012")
)
data_total <- data_total %>%
  select(-all_of(vars_to_delete))

data_total$year <- NULL

#all firms that were alive in 2012 are alive in 2014
sum(data_total$status_alive != data_total$status_alive_2012, na.rm = TRUE)


# create factors
data_total <- data_total %>%
  mutate(urban_m = factor(urban_m, levels = c(1,2,3)),
         ind2_cat = factor(ind2_cat, levels = sort(unique(data$ind2_cat))))


data$sales_mil
indicators <- c("comp_id", "year", "total_assets_bs", "personnel_exp_pl", "profit_loss_year_pl",
                "sales", "sales_mil", "sales_mil_log")

data_total <- data_total %>%
  filter(ind2_cat != 99)
data_total <- data_total %>%
  mutate(industry_type = ifelse(as.numeric(as.character(ind2_cat)) %in% c(20, 26, 27, 28, 29, 30, 32),
                                "manufacturing", "service"))


data_total$growth_rate <- exp(data_total$growth_ln_sales) - 1

data_total$diff_extra_profit_loss

data_total <- data_total %>%
  filter(diff_profit_loss_year < 200 & diff_extra_profit_loss > -100,
         diff_inc_bef_tax > -100 & diff_inc_bef_tax < 150,
         diff_share_eq > -100 & diff_share_eq < 100)

# List of growth variables in log differences
growth_vars <- c(
  "growth_ln_sales",
  "growth_ln_curr_assets",
  "growth_ln_curr_liab",
  "growth_ln_extra_inc",
  "growth_ln_fixed_assets",
  "growth_ln_intang_assets",
  "growth_ln_inventories",
  "growth_ln_liq_assets",
  "growth_ln_material_exp",
  "growth_ln_personnel_exp",
  "growth_ln_subscribed_cap",
  "growth_ln_tang_assets",
  "growth_ln_total_assets_bs",
  "growth_ln_amort",
  "growth_ln_extra_exp"
)

# Create dummy vars: 1 = positive growth, 0 = negative or no growth
for (var in growth_vars) {
  dummy_var <- paste0("flag_pos_", var)
  if (var %in% names(data_total)) {
    data_total[[dummy_var]] <- as.numeric(exp(data_total[[var]]) - 1 > 0)
  }
}

diff_vars <- c(
  "diff_extra_profit_loss",
  "diff_inc_bef_tax",
  "diff_profit_loss_year",
  "diff_share_eq"
)

# Loop through each diff variable and create a dummy for positive change
for (var in diff_vars) {
  dummy_var <- paste0("flag_pos_", var)
  if (var %in% names(data_total)) {
    data_total[[dummy_var]] <- as.numeric(data_total[[var]] > 0)
  }
}

# Dummy variables: 1 if increase > 100%, else 0
for (var in growth_vars) {
  dummy_var <- paste0("dummy_gt2x_", var)
  if (var %in% names(data_total)) {
    data_total[[dummy_var]] <- as.numeric(exp(data_total[[var]]) - 1 > 1)
  }
}

# Dummy variables: 1 if decrease > 50%, else 0
for (var in growth_vars) {
  dummy_var <- paste0("dummy_drop50_", var)
  if (var %in% names(data_total)) {
    data_total[[dummy_var]] <- as.numeric(exp(data_total[[var]]) - 1 < -0.5)
  }
}

# Recode dummy vars from numeric to categorical factor ("no", "yes")
dummy_vars <- names(data_total)[grepl("^flag_pos_|^dummy_gt2x_|^dummy_drop50_|^flag_pos_diff_", names(data_total))]

data_total <- data_total %>%
  mutate(across(all_of(dummy_vars), ~ factor(ifelse(. == 1, "yes", "no"), levels = c("no", "yes"))))



str(data_total)

write_csv(data_total,"data_cleaned.csv")
