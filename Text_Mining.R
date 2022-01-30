if(!require(tm)) install.packages("tm")
if(!require(wordcloud)) install.packages("wordcloud")
if(!require(plyr)) install.packages("plyr")
if(!require(dplyr)) install.packages("dplyr")
library(tm)
library(wordcloud)
library(plyr)
library(dplyr)

#Load the data
data <- read.csv('Reviews.csv', header=TRUE)

#Removing the Score of 3
data <- filter(data, Score != 3)

#Removing undesired columns
undesired <- c('ProductId', 'UserId', 'ProfileName', 'HelpfulnessDenominator', 'HelpfulnessNumerator',
               'Time', 'Summary', 'Id')

data <- data %>%
  select(-one_of(undesired))

#Encoding the Score as a binary variable
data$Score <- mapvalues(data$Score,
                        from=c(1,2,4,5),
                        to=c(0,0,1,1))

#As we can see, the distribution of the variable Score is imbalanced
table(data$Score)

#Export the preprocessed dataframe for the classifier
write.csv(data, 'Preprocessed.csv')

#Creation of the corpus positive and negative
pos <- filter(data, Score == 1)
neg <- filter(data, Score == 0)
corpus_pos <- Corpus(VectorSource(pos$Text))
corpus_neg <- Corpus(VectorSource(neg$Text))

#Basic operations on the two corpus
corpus_pos <- tm_map(corpus_pos, content_transformer(tolower))
corpus_pos <- tm_map(corpus_pos, removeNumbers)
corpus_pos <- tm_map(corpus_pos, removeWords, stopwords("english"))
corpus_pos <- tm_map(corpus_pos, removePunctuation)
corpus_pos <- tm_map(corpus_pos, stripWhitespace)

corpus_neg <- tm_map(corpus_neg, content_transformer(tolower))
corpus_neg <- tm_map(corpus_neg, removeNumbers)
corpus_neg <- tm_map(corpus_neg, removeWords, stopwords("english"))
corpus_neg <- tm_map(corpus_neg, removePunctuation)
corpus_neg <- tm_map(corpus_neg, stripWhitespace)

#Creation of the term document matrix
tdm_pos <- TermDocumentMatrix(corpus_pos)
tdm_neg <- TermDocumentMatrix(corpus_neg)

#Removing sparse words
sparse_pos <- removeSparseTerms(tdm_pos, 0.95)
sparse_neg <- removeSparseTerms(tdm_neg, 0.95)

#Wordcloud for positive reviews
m_pos <- as.matrix(sparse_pos)
v_pos <- sort(rowSums(m_pos), decreasing=TRUE)
d_pos <- data.frame(word=names(v_pos), freq=v_pos)
wordcloud(d_pos$word, d_pos$freq, random.order=FALSE, rot.per=0.3, scale=c(4,.5), max.words=101, colors=brewer.pal(8,'Dark2'))
title(main="Wordcloud - Positive Reviews", font.main=1, cex.main=1.5)

#Wordcloud for negative reviews
m_neg <- as.matrix(sparse_neg)
v_neg <- sort(rowSums(m_neg), decreasing=TRUE)
d_neg <- data.frame(word=names(v_neg), freq=v_neg)
wordcloud(d_neg$word, d_neg$freq, random.order=FALSE, rot.per=0.3, scale=c(4,.5), max.words=101, colors=brewer.pal(8,'Dark2'))
title(main="Wordcloud - Negative Reviews", font.main=1, cex.main=1.5)

#Correlation analysis
findAssocs(tdm_pos, terms = c("coffee","tea","dog","chips", "sugar"), corlimit = 0.25)
findAssocs(tdm_neg, terms = c("coffee","dog","tea","sugar", "chocolate"), corlimit = 0.25)

