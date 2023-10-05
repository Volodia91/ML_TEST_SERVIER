FROM python:3.6

# Install Miniconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add Miniconda to the PATH
ENV PATH=/opt/miniconda/bin:$PATH

# Update conda and create a new environment
RUN conda update -n base -c defaults conda && \
    conda create -y --name servier python=3.6 && \
    conda init bash && \
    echo "conda activate servier" >> ~/.bashrc

# Install rdkit library
RUN conda install -c conda-forge rdkit

# Set the working directory for Python project
WORKDIR /app

# Copy Python project files
COPY . /app

# Install any Python dependencies for your project
RUN pip install -r requirements.txt

# Your project-specific commands go here
CMD ["python", "app/setup.py"]

# Expose any necessary ports or define other settings as needed
EXPOSE 5000