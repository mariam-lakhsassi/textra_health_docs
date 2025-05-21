*Collecte de données:*

Nous avons utilisé des fichiers PDF et CSV existants sur les médicaments marocains et français:

* data.gov.ma
* snds.gouv.fr

*Prétraitement des données:*

Nous avons utilisé pdfplumber pour extraire le texte des PDF et langchain.text_splitter pour diviser les documents juridiques volumineux en morceaux plus petits et plus faciles à gérer.

*Création des embeddings:*

Chaque morceau a été converti en représentation vectorielle à l'aide du modèle d'embedding: mxbai-embed-large:latest.

*Création de la base de données vectorielle:*

Nous avons utilisé Chroma pour créer et persister la base de données vectorielle.

*Recherche et génération de réponse:*

Nous avons utilisé similarity_search de Chroma pour récupérer les morceaux de texte les plus pertinents de la base vectorielle pour la requête de l'utilisateur.

La réponse à la requête de l'utilisateur est générée à l'aide du modèle llama3:3b.
