# Load necessary library
library(tidyverse)
library(geomtextpath)
library(patchwork)
#library(ggpubr)

add_label_line <- function(y, x=0.1, text = "Value:",value=TRUE,...) {
  if (value){
    text=paste(text,y)
  }
  list(geom_hline(yintercept = y,...),
       annotate("text", x = x, y = y, label = text, vjust = -0.5)
  )
}

# Specify the file suffix to search for and folder name to ignore
file_suffix <- "CPMG.csv" # Suffix to match (here it's for CSV files)
ignore_folder <- "temp" # Name of the folder to ignore

# Get the current working directory
base_directory <- getwd()

# List all files in the current directory and subdirectories that match the suffix
all_files <- list.files(path = base_directory, pattern = file_suffix, full.names = TRUE, recursive = TRUE)

# Filter out files that are in the ignore_folder
filtered_files <- all_files[!grepl(paste0("/", ignore_folder, "/"), all_files)]

# Initialize an empty dataframe
combined_dataframe <- tibble()


# Read each file and append it to the combined dataframe

for (file in filtered_files) {
  temp_df <- read.csv(file)
  combined_dataframe <- bind_rows(combined_dataframe, temp_df)
}



#Convert sample names from code to publication type and set as levels
samplefrom <- c("RT","TT")
sampleto <- c("No Treatment", "Thermal Treatment")
combined_dataframe$Sample <- plyr::mapvalues(combined_dataframe$Sample,
                                             samplefrom,sampleto)
combined_dataframe$Sample <- factor(combined_dataframe$Sample, levels = sampleto)

#Convert mixing time to levels and proper unit publication type
CPMGperiodfrom <- c("1","0.5","0.25","0.1","0.025")
CPMGperiodto <- c("1 ms", "500 μs", "250 μs", "100 μs", "25 μs")
combined_dataframe$CPMG_period <- as.character(combined_dataframe$CPMG_period) %>%
                                  plyr::mapvalues(CPMGperiodfrom,CPMGperiodto)
combined_dataframe$CPMG_period <- factor(combined_dataframe$CPMG_period, levels = CPMGperiodto)

#Remove unnecessary data
#combined_dataframe <- combined_dataframe %>% filter(CPMG_period!='10 μs')


#Normalize intensity to the largest value of each sample
combined_dataframe <- combined_dataframe %>% group_by(Sample) %>%
                      mutate(Norm_Intensity=Intensity/max(Intensity)*100)

#Set plotting requirements
samples=c("No Treatment", "Thermal Treatment")
times=CPMGperiodto
levels<-10^(seq(from =-0.2, by =- 0.1, length.out=20))*100
# Basic plot

plot_cpmg <- function(data,sample,echo_time){
  curve= ggplot(filter(data,tau==echo_time & Sample==sample), aes(ppm,Norm_Intensity,colour=CPMG_period))+
    geom_line()+
    annotate("text", x=8, y=50, label= paste(sample)) +
    annotate("text", x=8, y=40, label= paste("Echo delay: ",echo_time,"ms")) +
    xlim(10,-2)+
    xlab("Chemical shift/ppm")+
    ylim(0,60)+
    ylab("Intensity")
  return(curve)
}
spectra<-lapply(samples,plot_cpmg,data=combined_dataframe,echo_time=1)


#col1 <- ggplot(aspect.ratio = 2) + annotate(geom = 'text', x=1, y=1, label=samples[1]) + theme_void() 

#col2 <- ggplot(aspect.ratio = 5) + annotate(geom = 'text', x=1, y=1, label=samples[2]) + theme_void()


final_plot <-  spectra[[1]]+spectra[[2]] + plot_layout(ncol=1, guides = "collect")

# final_plot <- final_plot + 
#               #plot_annotation(caption = paste('Levels = (',paste(signif(levels,digits=3),collapse=','),')'))+
#               plot_layout(ncol=1, guides = "collect", heights = c(1, 5,5,5))
#               

ggsave('cpmg.pdf',device = 'pdf',height=10,width=6)
ggsave('cpmg.png',device = 'png',height=10,width=6)

final_plot

# Check the structure of the combined dataframe
str(combined_dataframe)


#Convert from fiting components to physical components
fractionfrom <- c("tail","tail2","third","second","first")
fractionto <- c("Defects","Defects-2","HOC","DL","SL")

combined_dataframe$Fraction <- plyr::mapvalues(combined_dataframe$Fraction,
                                               fractionfrom,fractionto)





combined_dataframe$Fraction <- factor(combined_dataframe$Fraction, levels = fractionto)
combined_dataframe$Sample <- factor(combined_dataframe$Sample, levels = sampleto)


# build error bar data
error_bars = filter(combined_dataframe,Quantity=='A',is.element(Fraction,fractionto)) %>%
  arrange(Sample, desc(Fraction)) %>%
  # for each cyl group, calculate new value by cumulative sum
  group_by(Sample) %>%
  mutate(Value_new = cumsum(Value)) %>%
  mutate(Value = Value/max(Value_new),Error = Error/max(Value_new)) %>% 
  mutate(Value_new = Value_new/max(Value_new))%>%
  ungroup()


# Barplot with error bars
A_bar=ggplot(error_bars, aes(x = Sample, y = Value)) +
  geom_bar(stat = 'identity', aes(fill = Fraction)) +
  geom_errorbar(aes(ymax = Value_new + Error, ymin = Value_new - Error), 
                width = 0.1)+
  ylab("Fraction/%") + 
  theme(axis.text.x = element_text(angle = 90))

ggsave('A.pdf',device = 'pdf',width=9,height=6)
ggsave('A.png',device = 'png',width=9,height=6)
A_bar


#Plot Component fractions
A_plot= ggplot(filter(combined_dataframe,Quantity=='A',is.element(Fraction,fractionto)),
               aes(shape=Fraction,x=Sample,y=Value)) +
  geom_point() +
  geom_errorbar(aes(ymin=Value-Error,ymax=Value+Error),width=0.1)+
  ylab("Fraction/%") + 
  theme(axis.text.x = element_text(angle = 90))

ggsave('A_scatter.pdf',device = 'pdf',width=9,height=6)
ggsave('A_scatter.png',device = 'png',width=9,height=6)
A_plot

#Plot Dres values
Dres_plot= ggplot(filter(combined_dataframe,Quantity=='Dres'),
               aes(shape=Fraction,x=Sample,y=Value)) +
  geom_point()+
  geom_errorbar(aes(ymin=Value-Error,ymax=Value+Error),width=0.1)+
  #geom_crossbar(aes(ymin=Min,ymax=Max),width=0.2)+
  ylab("RDC /kHz")+ 
  scale_y_log10() + 
  add_label_line(y=c(0.06),x=3,text=c("Sakai:"),linetype='dashed')+ 
  theme(axis.text.x = element_text(angle = 90))

ggsave('Dres_log.pdf',device = 'pdf',width=9,height=6)
ggsave('Dres_log.png',device = 'png',width=9,height=6)

Dres_plot

#Plot Dres values
Dres_plot= ggplot(filter(combined_dataframe,Quantity=='Dres'),
                  aes(shape=Fraction,x=Sample,y=Value)) +
  geom_point()+
  geom_errorbar(aes(ymin=Value-Error,ymax=Value+Error),width=0.1)+
  #geom_crossbar(aes(ymin=Min,ymax=Max),width=0.2)+
  ylab("RDC /kHz")+
  #ylim(0.0, 0.3)+
  add_label_line(y=c(0.06),x=3,text=c("Sakai:"),linetype='dashed')+ 
  theme(axis.text.x = element_text(angle = 90))

ggsave('Dres_lin.pdf',device = 'pdf',width=9,height=6)
ggsave('Dres_lin.png',device = 'png',width=9,height=6)

Dres_plot

#Plot T2 values
T2_plot= ggplot(filter(combined_dataframe,Quantity=='T2'),
                  aes(shape=Fraction,x=Sample,y=Value)) +
  geom_point()+
  geom_errorbar(aes(ymin=Value-Error,ymax=Value+Error),width=0.1)+
  #geom_crossbar(aes(ymin=Min,ymax=Max),width=0.2)+
  ylab("T2 /ms")+
  scale_y_log10() +
  #ylim(0.0, 0.1)+
  #add_label_line(y=c(0.06),x=3,text=c("Sakai:"),linetype='dashed')+ 
  theme(axis.text.x = element_text(angle = 90))

ggsave('T2.pdf',device = 'pdf',width=9,height=6)
ggsave('T2.png',device = 'png',width=9,height=6)

T2_plot

#Plot beta values
beta_plot= ggplot(filter(combined_dataframe,Quantity=='beta'),
                aes(shape=Fraction,x=Sample,y=Value)) +
  geom_point()+
  geom_errorbar(aes(ymin=Value-Error,ymax=Value+Error),width=0.1)+
  #geom_crossbar(aes(ymin=Min,ymax=Max),width=0.2)+
  ylab("beta")+
  #ylim(0.0, 0.1)+
  #add_label_line(y=c(0.06),x=3,text=c("Sakai:"),linetype='dashed')+ 
  theme(axis.text.x = element_text(angle = 90))

ggsave('beta.pdf',device = 'pdf',width=9,height=6)
ggsave('beta.png',device = 'png',width=9,height=6)

beta_plot

#Plot sigma values
sigma_plot= ggplot(filter(combined_dataframe,Quantity=='sigma'),
                  aes(shape=Fraction,x=Sample,y=Value)) +
  geom_point()+
  geom_errorbar(aes(ymin=Value-Error,ymax=Value+Error),width=0.1)+
  #geom_crossbar(aes(ymin=Min,ymax=Max),width=0.2)+
  ylab("sigma")+
  #ylim(0.0, 0.1)+
  #add_label_line(y=c(0.06),x=3,text=c("Sakai:"),linetype='dashed')+ 
  theme(axis.text.x = element_text(angle = 90))

ggsave('sigma.pdf',device = 'pdf',width=9,height=6)
ggsave('sigma.png',device = 'png',width=9,height=6)

sigma_plot

# Optionally, save the combined dataframe to a CSV file
# write_csv(combined_dataframe, "combined_dataframe.csv")
