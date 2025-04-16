# Load necessary library
library(tidyverse)
library(baseline)
library(geomtextpath)
library(patchwork)


file_list<- function(file_suffix,ignore_folder="temp"){
  # Specify the file suffix to search for and folder name to ignore
  #file_suffix <-  Suffix to match (here it's for CSV files)
  #ignore_folder <- "temp"  Name of the folder to ignore
  
  # Get the current working directory
  base_directory <- getwd()
  
  # List all files in the current directory that match the suffix
  all_files <- list.files(path = base_directory, pattern = file_suffix, full.names = TRUE, recursive = FALSE)
  
  # Filter out files that are in the ignore_folder
  filtered_files <- all_files[!grepl(paste0("/", ignore_folder, "/"), all_files)]
  
  return(filtered_files)
}

#FUnction to add all the FID to a dataframe
FID_df<- function(file_list){
  # Initialize an empty dataframe
  
  combined_dataframe <- tibble()
  
  # Read each file and append it to the combined dataframe
  for (file in file_list) {
    temp_df <- read_tsv(file,col_names = c("time","real","imaginary")) %>% 
      mutate(delay_index=as.numeric(gsub(".*?([0-9]+)$","\\1", file)))
    if (!("imaginary" %in% names(temp_df))){
      temp_df = mutate(temp_df,imaginary=0)
    }
    combined_dataframe <- bind_rows(combined_dataframe, temp_df)
  }
  return(combined_dataframe)
}

FID_plot<-function(FID){
  ignore_points=10
  fid_plt= ggplot(filter(FID,row_number() %% ignore_points == 0),aes(x=time,y=real,color=delay_index))+
    geom_point()+
    #geom_point(aes(x=time,y=imaginary))+
    annotate("text",label=paste("Every ",ignore_points," points plotted for clarity"),x=Inf,y=Inf, vjust = 2, hjust = 1)+
    xlab("Time/ms")
  return(fid_plt)
}

DQ_files <- file_list("BP_303.txtdq")
DQ_FID <- FID_df(DQ_files)

DQ_plot = FID_plot(DQ_FID)
DQ_plot

ref_files <- file_list("BP_303.txtref")
ref_FID <- FID_df(ref_files)

ref_plot = FID_plot(ref_FID)
ref_plot

