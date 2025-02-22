library(AdhereR)
library(dplyr)

# Select only the five relevant columns
med_events_subset <- med.events %>% select(PATIENT_ID, DATE, PERDAY, CATEGORY, DURATION)

# Save the dataset as a CSV file
write.csv(med_events_subset, "med_events.csv", row.names = FALSE)
