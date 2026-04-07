FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

WORKDIR /app

COPY requirements.txt .

# install dependencies + pin versions
RUN pip install --no-cache-dir --ignore-installed sympy==1.13.3 \
 && pip install --no-cache-dir git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@2019.1.0 \
 && pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# patch fenics_adjoint in-place (debug-safe)
RUN set -eux; \
    cd /usr/lib/python3/dist-packages/fenics_adjoint; \
    grep -rlE "^\s*import ufl(\s|$)" . | xargs -r sed -i 's/^\s*import ufl\s*$/import ufl_legacy as ufl/g'; \
    grep -rl "import ufl.algorithms" . | xargs -r sed -i '/import ufl\.algorithms/d'

# start in bash
CMD ["bash"]
