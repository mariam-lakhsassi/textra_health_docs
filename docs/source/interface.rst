TextraHealth Web Interface
=========================

Overview
--------

The TextraHealth web interface is a user-friendly application built with Streamlit that allows users to upload medical prescriptions, receive detailed analysis reports, and interact with a virtual assistant (chatbot) for further information about prescribed medications.

Main Features
-------------

- **Prescription Upload:**  
  Users can upload images of their medical prescriptions in JPG, JPEG format (up to 200MB per file).

  .. image:: ../images/1.jpeg
     :width: 600
     :align: center
     :alt: Upload prescription interface

- **Automated Prescription Analysis:**  
  After uploading, the system automatically analyzes the prescription using OCR and AI models to extract key information such as:
  
  - List of prescribed medications
  - Dosage, frequency, and duration
  - Important usage instructions
  - Side effects and warnings

  .. image:: ../images/2.jpeg
     :width: 600
     :align: center
     :alt: Prescription analysis

- **Detailed Report Generation:**  
  The extracted information is presented in a clear, structured report. The report includes:
  
  - Medication details
  - Important advice (e.g., take with water, avoid alcohol)
  - Warnings and precautions

  .. image:: ../images/3.jpeg
     :width: 600
     :align: center
     :alt: Analysis report

- **Medical Chatbot Assistant:**  
  Users can ask follow-up questions in natural language (e.g., "Quelles sont les effets secondaires du médicament prescrit dans cette ordonnance?"). The chatbot provides context-aware answers based on the prescription and medical knowledge.

  .. image:: ../images/4.jpeg
     :width: 600
     :align: center
     :alt: Chatbot question

  .. image:: ../images/5.jpeg
     :width: 600
     :align: center
     :alt: Chatbot answer

- **Disclaimer:**  
  The interface clearly states that it does not replace professional medical advice.

  .. image:: ../images/6.jpeg
     :width: 600
     :align: center
     :alt: Disclaimer

User Workflow
-------------

1. **Upload Prescription**
   
   - Click on "Browse files" or drag and drop your prescription image into the upload area.
   - Supported formats: JPG, JPEG.

   .. image:: ../images/1.jpeg
      :width: 600
      :align: center

2. **Analyze Prescription**
   
   - Click "Analyser l'ordonnance".
   - The system processes the image and displays a summary of the prescription and a detailed analysis report.

   .. image:: ../images/2.jpeg
      :width: 600
      :align: center

3. **View Analysis Report**
   
   - The report includes prescribed medications, dosages, usage instructions, and warnings.
   - Example:
   
     .. code-block:: text

        Médicaments Prescrits:
        - Azithromycin (dosage : 20 mg/smL, fréquence : non spécifiée, durée : non spécifiée) - c'est un antibiotique macrolide utilisé pour traiter diverses infections bactériennes.

        Conseils Importants:
        - Prendre avec de l'eau : cela indique que le patient doit prendre le médicament avec une quantité suffisante d'eau pour faciliter la digestion et la biodisponibilité du médicament.
        - Éviter l'alcool : cela peut être précautionnaire, car certains médicaments peuvent interagir avec l'alcool ou affecter ses effets.

   .. image:: ../images/3.jpeg
      :width: 600
      :align: center

4. **Ask the Chatbot**
   
   - Switch to the "Chatbot Médical" tab or use the chat box.
   - Ask questions such as:
     - "Quelles sont les effets secondaires du médicament prescrit dans cette ordonnance ?"
     - "À quoi sert ce médicament ?"
   - The chatbot responds with relevant, context-aware information.

   .. image:: ../images/4.jpeg
      :width: 600
      :align: center

   .. image:: ../images/5.jpeg
      :width: 600
      :align: center

5. **Disclaimer**
   
   - A warning is displayed: "Cet outil ne remplace pas un avis médical professionnel."

   .. image:: ../images/6.jpeg
      :width: 600
      :align: center

Technical Notes
---------------

- **Built with Streamlit:**  
  The interface leverages Streamlit for rapid development and interactive UI.
- **Backend Integration:**  
  The analysis and chatbot features are powered by OCR, NLP, and LLM models described in the pipeline documentation.

Summary
-------

TextraHealth's interface streamlines the process of understanding medical prescriptions by combining automated analysis and an intelligent chatbot, making medical information more accessible and understandable for patients and healthcare professionals.