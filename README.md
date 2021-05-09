## Aventuras con texto (scroll down for Enlish)

Aquí tienes unos notebooks que he creado en Jupyter (tanto en español como en inglés) para acompañar las clases que doy en un máster de Inteligencia Artificial sobre los últimos desarrollos en PNL (procesamiento del lenguaje natural o NLP en las siglas inglesas) con redes neuronales.

Se dice que 2018 fue el año "ImageNet" para texto. Se refiere a los avances en el reconocimiento de imágenes y, en particular, a transfer learning. Es decir, la posibilidad de entrenar un modelo grande, computacionalmente costoso con un conjunto de datos generales, y poder "tunear" este modelo para una tarea específica (por ejemplo, diferenciar entre perros y gatos). Hasta hace poco, no era factible aplicar transfer learning a modelos basados en texto (o PNL).

* [Referencias](https://github.com/teticio/aventuras-con-textos/blob/master/Referencias.ipynb). Una lista de enlaces a todos los trabajos académicos relevantes.

* [Clasificación de texto con modelos de última generación](https://github.com/teticio/aventuras-con-textos/blob/master/Clasificacion_de_texto_con_modelos_de_ultima_generacion.ipynb). Una introducción y comparación entre los modelos Word2Vec, ELMo, BERT y XLNet para clasificar reseñas de películas IMDB como positivas o negativas.

* [Atención](https://github.com/teticio/aventuras-con-textos/blob/master/Atencion.ipynb). Una mirada en profundidad al mecanismo de atención utilizado en el Transformador, el componente principal de BERT, partiendo de un simple modelo Vec2Vec para traducir del inglés al español.

* [BERT entiende](https://github.com/teticio/aventuras-con-textos/blob/master/BERT_entiende.ipynb). Aquí usamos un modelo BERT que se ha tuneado con la tarea SQuAD (conjunto de datos de respuesta a preguntas de Stanford) para responder a preguntas de comprensión de lectura sobre un libro de Harry Potter.

* [BERT predice](https://github.com/teticio/aventuras-con-textos/blob/master/BERT_predice.ipynb). BERT está entrenado para poder completar las palabras que faltan en una frase.

* [Bertle](https://github.com/teticio/aventuras-con-textos/blob/master/Bertle.ipynb). Un motor de búsqueda semántico que utiliza embeddings a nivel de frase de BERT para encontrar artículos relevantes de Stack Overflow.

* [Dr. BERT](https://github.com/teticio/aventuras-con-textos/blob/master/Dr_Bert.ipynb). Un psicoanalista inspirado en Eliza y entrenado usando las transcripciones del Dr. Carl Rogers.

* [Modelos de lenguaje](https://github.com/teticio/aventuras-con-textos/blob/master/Modelos_de_lenguaje.ipynb). Un modelo de lenguaje es una función que estima la probabilidad de la siguiente palabra (o token) condicionada en el texto que la precede. Aquí vamos a utilizar el modelo de lenguaje GPT-2 para predecir la continuación de una frase y destacar construcciones poco probables.

* [Generación de texto con modelos de última generación](https://github.com/teticio/aventuras-con-textos/blob/master/Modelos_generativos_de_texto_de_ultima_generacion.ipynb). Los modelos de lenguaje XLNet y GPT-2 se utilizan para generar una prosa aleatoria, desde escribir capítulos de Game of Thrones hasta generar tweets al estilo de Donald Trump.

* [Opiniones de Amazon](https://github.com/teticio/aventuras-con-textos/blob/master/Amazon_Opiniones.ipynb). Una competición de estilo Kaggle para usar lo que has aprendido para crear un modelo para clasificar las reseñas de Amazon como negativas o positivas. Los desafíos adicionales surgen de tener un conjunto de datos muy pequeño y desequilibrado en español.

* [GPT-2](https://github.com/teticio/aventuras-con-textos/blob/master/GPT2.py). Un script de Python que utiliza la implementación PyTorch de Hugging Face para generar texto con el modelo GPT-2 de 1.500 millones de parámetros lanzado por OpenAI en noviembre de 2019.

Todos los notebooks se pueden ejecutar en [Google Colab] (https://colab.research.google.com/github/teticio/aventuras-con-textos). Si quieres acceder a los checkpoints previamente entrenados (aproximadamente 10 Gb) en Google Colab, envía este [enlace] (https://drive.google.com/drive/folders/1QB6Pr5U1AUQMtk-GzHLa6ijJXiagsS4r?usp=sharing) a tu cuenta de Gmail y guarda el directorio con el nombre "checkpoints" en tu Google Drive en un subdirectorio (que puede ser necesario crear) llamado "Cuadernos Colab". Ten en cuenta que los notebooks actualmente funcionan con TensorFlow 1.14 y los puntos de control son compatibles con Keras 2.2.4. En Google Colab se pueden con !pip uninstall -y tensorflow, !pip install --upgrade tensorflow-gpu==1.14 y !pip install --upgrade keras==2.2.4.

## Adventures with text

This is a set of Jupyter notebooks I have created (in both Spanish and English) to accompany classes I give in Masters in Artificial Intelligence on the latest developments in end-to-end NLP (Natural Language Processing) with neural networks.

Some people say that 2018 was the "ImageNet" year for text. By this they are referring to the breakthroughs in image recognition and, in particular, transfer learning. That is to say, the possibility of training a large, computationally expensive model on a general data set, and being able to "fine-tune" this model for a specific task (for example, to tell the difference between dogs and cats). Up until recently, it has not been feasible to apply transfer learning to text based (or NLP) models.

* [References](https://github.com/teticio/aventuras-con-textos/blob/master/Referencias.ipynb). A list of links to all the relevant academic papers.

* [Classification of text with cutting edge models](https://github.com/teticio/aventuras-con-textos/blob/master/Clasificacion_de_texto_con_modelos_de_ultima_generacion.ipynb). An introduction to and comparison of Word2Vec, ELMo, BERT and XLNet models for classifying IMDB movie reviews as either positive or negative.

* [Attention](https://github.com/teticio/aventuras-con-textos/blob/master/Atencion.ipynb). A deep dive into the attention mechanism used in the Transformer - the main building block of BERT - starting from a simple Vec2Vec model to translate from English to Spanish.

* [BERT understands](https://github.com/teticio/aventuras-con-textos/blob/master/BERT_entiende.ipynb). Here we use a BERT model that has been fine-tuned on the SQuAD (Stanford Question Answering Dataset) to answer reading comprehension questions about a Harry Potter book.

* [BERT predicts](https://github.com/teticio/aventuras-con-textos/blob/master/BERT_predice.ipynb). BERT is trained to be able to fill in the missing words in a sentence.

* [Bertle](https://github.com/teticio/aventuras-con-textos/blob/master/Bertle.ipynb). A semantic search engine that uses BERT sentence embeddings to find relevant articles from Stack Overflow.

* [Dr. BERT](https://github.com/teticio/aventuras-con-textos/blob/master/Dr_Bert.ipynb). A psychoanalyst inspired by Eliza and trained using the transcripts of Dr Carl Rogers.

* [Language models](https://github.com/teticio/aventuras-con-textos/blob/master/Modelos_de_lenguaje.ipynb). A language model is a function that estimates the probability of the next word (or token) conditioned on the text that precedes it. Here we are going to use the GPT-2 language model to predict the continuation of a sentence and to draw attention to unlikely constructions.

* [Text generation with cutting edge models](https://github.com/teticio/aventuras-con-textos/blob/master/Modelos_generativos_de_texto_de_ultima_generacion.ipynb). Language Models XLNet and GPT-2 are used to generate random prose, from writing chapters of Game of Thrones to generating tweets in the style of Donald Trump.

* [Amazon opinions](https://github.com/teticio/aventuras-con-textos/blob/master/Amazon_Opiniones.ipynb). A Kaggle style competition to use what you have learned to create a model to classify Amazon reviews as either negative or positive. Extra challenges arise from having a very small, unbalanced data set in Spanish.

* [GPT-2](https://github.com/teticio/aventuras-con-textos/blob/master/GPT2.py). A Python script using Hugging Face's PyTorch implementation to generate text with the 1.5 billion parameter GPT-2 model released by OpenAI in November 2019.

All the notebooks can be run on Google Colab. If you want to access the pre-trained checkpoints (approximately 10 Gb) on Google Colab, send this link to your Gmail account and save the directory with the name "checkpoints" to your Google Drive in a subdirectory (which you may need to create) called "Colab Notebooks". Note that the notebooks currently work with TensorFlow 1.14 and the checkpoints are compatible with Keras 2.2.4. On Google Colab you can install these with !pip uninstall -y tensorflow, !pip install --upgrade tensorflow-gpu==1.14 and !pip install --upgrade keras==2.2.4.
