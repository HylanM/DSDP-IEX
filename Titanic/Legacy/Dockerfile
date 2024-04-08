FROM jupyter/datascience-notebook:latest
WORKDIR /home/jovyan/work
COPY . /home/jovyan/work/
EXPOSE 8888
CMD ["start-notebook.sh", "--NotebookApp.token=''"]

# docker build -t titanic-project .
# docker run -p 8888:8888 --name titanic2 titanic-project