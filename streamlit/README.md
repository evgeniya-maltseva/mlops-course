This is the folder with the streamlit code for the pairings comparison app.
It can be run:
1. throught terminal locally from the folder streamlit/app 
```
streamlit run pairings.py
```
2. by creating docker container
    - Build an image
    ```
    docker build -t airport .
    ```

    - Run Docker container
    ```
    docker run -p 8501:8501 airport
    ```
3. the app also saved in streamlit cloud and can be access by this link 

https://compairings.streamlit.app/

Note: The most actual up-to-date version of the app is in st_airport branch of git repository