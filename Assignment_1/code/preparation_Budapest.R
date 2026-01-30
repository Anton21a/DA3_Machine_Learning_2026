library(tidyverse)
library(modelsummary)
library(dplyr)
rm(list=ls())

setwd("C:/Users/user/Desktop/MA2y/Data_Analysis_3/DA3_Machine_Learning_2026/Assignment_1/data")

data2025q2 <- read.csv("Budapest2025q2_cleaned.csv")
data2025q1 <- read.csv("Budapest2025q1_cleaned.csv")

data <- data2025q1 %>%
  dplyr::group_by(property_type) %>%
  dplyr::summarise(n = dplyr::n_distinct(X), .groups = "drop")

table(data$property_type)

data2025q1 <- data2025q1 %>%
  filter(property_type %in% c("Entire rental unit", "Entire condo", 
                              "Private room in rental unit")) %>%
  mutate(
    property_type = recode(property_type,
                           "Entire rental unit" = "Rental Unit",
                           "Entire condo" = "Condo",
                           "Private room in rental unit" = "Private Room"),
    property_type = factor(property_type))

data2025q2 <- data2025q2 %>%
  filter(property_type %in% c("Entire rental unit", "Entire condo", 
                              "Private room in rental unit")) %>%
  mutate(
    property_type = recode(property_type,
                           "Entire rental unit" = "Rental Unit",
                           "Entire condo" = "Condo",
                           "Private room in rental unit" = "Private Room"),
    property_type = factor(property_type))

data2025q1 <- data2025q1 %>%
  mutate(f_room_type = factor(room_type))
data2025q2 <- data2025q2 %>%
  mutate(f_room_type = factor(room_type))


data2025q1$f_room_type2 <- factor(ifelse(data2025q1$f_room_type== "Entire home/apt", "Entire/Apt",
                                       ifelse(data2025q1$f_room_type== "Private room", "Private", ".")))

data2025q2$f_room_type2 <- factor(ifelse(data2025q2$f_room_type== "Entire home/apt", "Entire/Apt",
                                       ifelse(data2025q2$f_room_type== "Private room", "Private", ".")))

## Create Numerical variables
data2025q1 <- data2025q1 %>%
  mutate(
    Hung_Foriant_price_day = price,
    p_host_response_rate = ifelse(host_response_rate == "N/A", -1,
                                  as.numeric(gsub("%", "", host_response_rate))),
    p_host_acceptance_rate = ifelse(host_acceptance_rate == "N/A", -1,
                                    as.numeric(gsub("%", "", host_acceptance_rate)))
  )

numericals <- c("accommodates","bathrooms","review_scores_rating","number_of_reviews",
                "reviews_per_month","minimum_nights","beds")
data2025q1 <- data2025q1 %>%
  mutate_at(vars(numericals), funs("n"=as.numeric))

data2025q2 <- data2025q2 %>%
  mutate(
    Czech_koruna_price_day = price,
    p_host_response_rate = ifelse(host_response_rate == "N/A", -1,
                                  as.numeric(gsub("%", "", host_response_rate))),
    p_host_acceptance_rate = ifelse(host_acceptance_rate == "N/A", -1,
                                    as.numeric(gsub("%", "", host_acceptance_rate)))
  )

numericals <- c("accommodates","bathrooms","review_scores_rating","number_of_reviews",
                "reviews_per_month","minimum_nights","beds")
data2025q2 <- data2025q2 %>%
  mutate_at(vars(numericals), funs("n"=as.numeric))


#create days since first review
data2025q1 <- data2025q1 %>%
  mutate(
    n_days_since = as.numeric(as.Date(calendar_last_scraped,format="%Y-%m-%d") -
                                as.Date(first_review ,format="%Y-%m-%d")))

data2025q2 <- data2025q2 %>%
  mutate(
    n_days_since = as.numeric(as.Date(calendar_last_scraped,format="%Y-%m-%d") -
                                as.Date(first_review ,format="%Y-%m-%d")))

data2025q1 %>%
  filter(price < 200000) %>%
  ggplot(aes(x = price)) +
  geom_histogram(binwidth = 250, fill = "darkgreen", color = "blue") +
  labs(title = "Distribution of Nightly Prices (Below 10,000 CZK)",
       x = "Price",
       y = "Number of Listings") +
  theme_minimal()

data2025q1 <- data2025q1 %>%
  mutate(ln_price = log(price))
data2025q1 <- data2025q1 %>%
  filter(price < 80000)

data2025q2 %>%
  filter(price < 100000) %>%
  ggplot(aes(x = price)) +
  geom_histogram(binwidth = 250, fill = "darkgreen", color = "blue") +
  labs(title = "Distribution of Nightly Prices (Below 10,000 CZK)",
       x = "Price",
       y = "Number of Listings") +
  theme_minimal()

data2025q2 <- data2025q2 %>%
  mutate(ln_price = log(price))
data2025q2 <- data2025q2 %>%
  filter(price < 50000)

data2025q1 <- data2025q1 %>%
  mutate(accommodates2=accommodates^2, 
         ln_accommodates=log(accommodates) ,
         ln_accommodates2=log(accommodates)^2,
         ln_beds = log(beds),
         ln_number_of_reviews = log(number_of_reviews+1)
  )

data2025q2 <- data2025q2 %>%
  mutate(accommodates2=accommodates^2, 
         ln_accommodates=log(accommodates) ,
         ln_accommodates2=log(accommodates)^2,
         ln_beds = log(beds),
         ln_number_of_reviews = log(number_of_reviews+1)
  )

# Pool accomodations with 0,1,2,10 bathrooms
data2025q1 <- data2025q1 %>%
  mutate(f_bathroom = cut(bathrooms, c(0,1,2,10), labels=c(0,1,2), right = F) )

# Pool num of reviews to 3 categories: none, 1-51 and >51
data2025q1$number_of_reviews_n <- as.numeric(data2025q1$number_of_reviews_n)
data2025q1 <- data2025q1 %>%
  mutate(f_number_of_reviews = cut(
    number_of_reviews_n,
    breaks = c(0, 1, 51, max(number_of_reviews_n, na.rm = TRUE)),
    labels = c(0, 1, 2),
    right = FALSE
  ))

# Pool and categorize the number of minimum nights: 1,2,3, 3+
data2025q1$minimum_nights <- as.numeric(data2025q1$minimum_nights)
data2025q1 <- data2025q1 %>%
  mutate(f_minimum_nights = cut(
    minimum_nights,
    breaks = c(0, 2, 3, max(minimum_nights, na.rm = TRUE)),
    labels = c(0, 1, 2),
    right = FALSE
  ))

# Change Infinite values with NaNs
for (j in 1:ncol(data2025q1) ) data.table::set(data2025q1,
                                             which(is.infinite(data2025q1[[j]])), j, NA)

data2025q2 <- data2025q2 %>%
  mutate(f_bathroom = cut(bathrooms, c(0,1,2,10), labels=c(0,1,2), right = F) )

data2025q2$number_of_reviews_n <- as.numeric(data2025q2$number_of_reviews_n)
data2025q2 <- data2025q2 %>%
  mutate(f_number_of_reviews = cut(
    number_of_reviews_n,
    breaks = c(0, 1, 51, max(number_of_reviews_n, na.rm = TRUE)),
    labels = c(0, 1, 2),
    right = FALSE
  ))

data2025q2$minimum_nights <- as.numeric(data2025q2$minimum_nights)
data2025q2 <- data2025q2 %>%
  mutate(f_minimum_nights = cut(
    minimum_nights,
    breaks = c(0, 2, 3, max(minimum_nights, na.rm = TRUE)),
    labels = c(0, 1, 2),
    right = FALSE
  ))

for (j in 1:ncol(data2025q2) ) data.table::set(data2025q2,
                                             which(is.infinite(data2025q2[[j]])), j, NA)

to_filter <- sapply(data2025q1, function(x) sum(is.na(x)))
to_filter[to_filter > 0]

vars_to_drop <- c(
  "host_is_superhost", "neighbourhood_group_cleansed",
  "calendar_updated", "number_of_reviews",
  "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
  "review_scores_checkin", "review_scores_communication", "review_scores_location",
  "review_scores_value", "license", "reviews_per_month",
  "review_scores_rating_n", "reviews_per_month_n", "n_days_since",
  "ln_beds", "f_number_of_reviews"
)
data2025q1 <- data2025q1 %>%
  select(-all_of(vars_to_drop))

to_filter <- sapply(data2025q2, function(x) sum(is.na(x)))
to_filter[to_filter > 0]

data2025q2 <- data2025q2 %>%
  select(-all_of(vars_to_drop))

data2025q1 <- data2025q1 %>%
  mutate(
    bathrooms =  ifelse(is.na(bathrooms), median(bathrooms, na.rm = T), bathrooms),
    bedrooms =  ifelse(is.na(bedrooms), median(bedrooms, na.rm = T), bedrooms),
    beds_n = ifelse(is.na(beds_n), accommodates, beds_n),
    f_bathroom=ifelse(is.na(f_bathroom),1, f_bathroom),
    f_minimum_nights=ifelse(is.na(f_minimum_nights),1, f_minimum_nights)
  )

data2025q2 <- data2025q2 %>%
  mutate(
    bathrooms =  ifelse(is.na(bathrooms), median(bathrooms, na.rm = T), bathrooms),
    bedrooms =  ifelse(is.na(bedrooms), median(bedrooms, na.rm = T), bedrooms),
    beds_n = ifelse(is.na(beds_n), accommodates, beds_n),
    f_bathroom=ifelse(is.na(f_bathroom),1, f_bathroom),
    f_minimum_nights=ifelse(is.na(f_minimum_nights),1, f_minimum_nights)
  )

data2025q1 <- na.omit(data2025q1)
data2025q2 <- na.omit(data2025q2)

write.csv(data2025q1, 'Budapest2025q1_final.csv')
write.csv(data2025q2, 'Budapest2025q2_final.csv')
