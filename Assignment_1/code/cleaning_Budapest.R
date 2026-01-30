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

data2025q1 <- read.csv("Budapest2025q1.csv")
data2025q2 <- read.csv("Budapest2025q2.csv")

drops <- c("host_thumbnail_url","host_picture_url","listing_url","picture_url","host_url","last_scraped","description", "neighborhood_overview", "host_about", "host_response_time", "name", "host_location")

data2025q1<-data2025q1[ , !(names(data2025q1) %in% drops)]
data2025q2<-data2025q2[ , !(names(data2025q2) %in% drops)]

#drop broken lines - where id is not a character of numbers
data2025q1$junk<-grepl("[[:alpha:]]", data2025q1$id)
data2025q1<-subset(data2025q1,data2025q1$junk==FALSE)
data2025q1<-data2025q1[1:ncol(data2025q1)-1]

data2025q2$junk<-grepl("[[:alpha:]]", data2025q2$id)
data2025q2<-subset(data2025q2,data2025q2$junk==FALSE)
data2025q2<-data2025q2[1:ncol(data2025q2)-1]

#the class and type of each columns
sapply(data2025q1, class)
sapply(data2025q1, typeof)

for (perc in c("host_response_rate","host_acceptance_rate")){
  data2025q1[[perc]]<-gsub("%","",as.character(data2025q1[[perc]]))
}

data2025q1$price <- gsub("[\\$,]", "", data2025q1$price)        
data2025q1$price <- as.numeric(data2025q1$price)   

for (perc in c("host_response_rate","host_acceptance_rate")){
  data2025q2[[perc]]<-gsub("%","",as.character(data2025q2[[perc]]))
}

#remove dollar signs from price variables
data2025q2$price <- gsub("[\\$,]", "", data2025q2$price)        
data2025q2$price <- as.numeric(data2025q2$price)

#format binary variables
for (binary in c("host_is_superhost","host_has_profile_pic","host_identity_verified",
                 "instant_bookable")){
  data2025q1[[binary]][data2025q1[[binary]]=="f"] <- 0
  data2025q1[[binary]][data2025q1[[binary]]=="t"] <- 1
}

for (binary in c("host_is_superhost","host_has_profile_pic","host_identity_verified",
                 "instant_bookable")){
  data2025q2[[binary]][data2025q2[[binary]]=="f"] <- 0
  data2025q2[[binary]][data2025q2[[binary]]=="t"] <- 1
}

data2025q1$amenities <- gsub("[\\{\\}\"]", "", data2025q1$amenities)
data2025q1$amenities <- tolower(data2025q1$amenities)
data2025q1$amenities <- trimws(data2025q1$amenities)
data2025q1$amenities <- strsplit(data2025q1$amenities, ",")
data2025q1$amenities <- lapply(data2025q1$amenities, trimws)

amenity_counts <- sort(table(unlist(data2025q1$amenities)), decreasing = TRUE)

top_n <- 20  
top_amenities <- names(amenity_counts)[1:top_n]

top_dummies <- as.data.frame(do.call(rbind, lapply(data2025q1$amenities, function(x) {
  as.integer(top_amenities %in% x)
})))

colnames(top_dummies) <- make.names(top_amenities)
data2025q1 <- cbind(data2025q1, top_dummies)


data2025q2$amenities <- gsub("[\\{\\}\"]", "", data2025q2$amenities)
data2025q2$amenities <- tolower(data2025q2$amenities)
data2025q2$amenities <- trimws(data2025q2$amenities)
data2025q2$amenities <- strsplit(data2025q2$amenities, ",")
data2025q2$amenities <- lapply(data2025q2$amenities, trimws)

amenity_counts <- sort(table(unlist(data2025q2$amenities)), decreasing = TRUE)

top_n <- 20  
top_amenities <- names(amenity_counts)[1:top_n]

top_dummies <- as.data.frame(do.call(rbind, lapply(data2025q2$amenities, function(x) {
  as.integer(top_amenities %in% x)
})))

colnames(top_dummies) <- make.names(top_amenities)
data2025q2 <- cbind(data2025q2, top_dummies)

data2025q1$amenities <- NULL
data2025q2$amenities <- NULL

write.csv(data2025q1, 'Budapest2025q1_cleaned.csv')
write.csv(data2025q2, 'Budapest2025q2_cleaned.csv')










































