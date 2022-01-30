library(keras)
library(tensorflow)

#Load the data
df <- read.csv('Preprocessed.csv', header = TRUE)
df <- df[,-1]

#Dividing it into train/test sets
training_id <- sample.int(nrow(df), size = nrow(df)*0.8)
training <- df[training_id,]
testing <- df[-training_id,]

table(training$Score)
table(testing$Score)

#Set the text vectorization layer
num_words <- 10000
max_length <- 50
text_vectorization <- layer_text_vectorization(
  max_tokens = num_words, 
  output_sequence_length = max_length
)

#Fit the layer to our text reviews
text_vectorization %>% 
  adapt(df$Text)

#This is our vocabulary 
get_vocabulary(text_vectorization)

#Creating the model (several layers)
text_vectorization(matrix(df$Text[1], ncol = 1))

input <- layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)

#Choosing the methods and metrics for model optimization
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = tf$metrics$AUC()
)

#Training the model
history <- model %>% fit(
  training$Text,
  as.numeric(training$Score == 1),
  epochs = 10,
  batch_size = 512,
  validation_split = 0.2,
  verbose=2
)

#Evaluating the model
results <- model %>% evaluate(testing$Text, as.numeric(testing$Score == 1), verbose = 0)
results

#Plotting the results
plot(history)
