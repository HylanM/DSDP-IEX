FROM jupyter/datascience-notebook:latest
WORKDIR /home/jovyan/work
COPY . /home/jovyan/work/
EXPOSE 8888
CMD ["start-notebook.sh", "--NotebookApp.token=''"]

# docker build -t housing-project .
# docker run -p 8888:8888 --name housing1 housing-project