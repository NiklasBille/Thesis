FROM dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

RUN pip install rdkit-pypi==2022.9.3

RUN pip3 install ipykernel

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    pip cache purge