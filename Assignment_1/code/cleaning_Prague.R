library(caret)
library(tidyverse)
library(skimr)
library(ggthemes)
library(gridExtra)
library(lattice)
library(glmnet)
library(rpart)
library(rattle)
library(rpart.plot)
library(xtable)
library(Hmisc)
library(modelsummary)


#-----------------------------------------------------------------------------------------
rm(list=ls())

setwd("C:/Users/user/Desktop/MA2y/Data_Analysis_3/DA3_Machine_Learning_2026/Assignment_1/data")

data2025 <- read.csv("Prague2025q2.csv")


drops <- c("host_thumbnail_url","host_picture_url","listing_url","picture_url","host_url","last_scraped","description", "neighborhood_overview", "host_about", "host_response_time", "name", "host_location")

data2025<-data2025[ , !(names(data2025) %in% drops)]

#drop broken lines - where id is not a character of numbers
data2025$junk<-grepl("[[:alpha:]]", data2025$id)
data2025<-subset(data2025,data2025$junk==FALSE)
data2025<-data2025[1:ncol(data2025)-1]



for (perc in c("host_response_rate","host_acceptance_rate")){
  data2025[[perc]]<-gsub("%","",as.character(data2025[[perc]]))
}

#remove dollar signs from price variables
data2025$price <- gsub("[\\$,]", "", data2025$price)        
data2025$price <- as.numeric(data2025$price)

#format binary variables
for (binary in c("host_is_superhost","host_has_profile_pic","host_identity_verified",
                 "instant_bookable")){
  data2025[[binary]][data2025[[binary]]=="f"] <- 0
  data2025[[binary]][data2025[[binary]]=="t"] <- 1
}

data2025$amenities <- gsub("[\\{\\}\"]", "", data2025$amenities)
data2025$amenities <- tolower(data2025$amenities)
data2025$amenities <- trimws(data2025$amenities)
data2025$amenities <- strsplit(data2025$amenities, ",")
data2025$amenities <- lapply(data2025$amenities, trimws)

amenity_counts <- sort(table(unlist(data2025$amenities)), decreasing = TRUE)

top_n <- 20  
top_amenities <- names(amenity_counts)[1:top_n]

top_dummies <- as.data.frame(do.call(rbind, lapply(data2025$amenities, function(x) {
  as.integer(top_amenities %in% x)
})))

colnames(top_dummies) <- make.names(top_amenities)
data2025 <- cbind(data2025, top_dummies)

data2025$amenities <- NULL

write.csv(data2025, 'Prague2025q2_cleaned.csv')










































