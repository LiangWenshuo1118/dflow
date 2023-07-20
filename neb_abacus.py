import numpy as np
import os, time, glob
from pathlib import Path
from typing import List
from dflow import(
    Step,
    Steps,
    Workflow,
    upload_artifact,
    download_artifact
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate, 
    Parameter
)

from dflow.plugins.bohrium import BohriumContext, BohriumExecutor
from dflow import config, s3_config
config["host"] = "https://workflow.test.dp.tech"
s3_config["endpoint"] = "39.106.93.187:30900"
config["k8s_api_server"] = "https://60.205.59.4:6443"
config["token"] = "eyJhbGciOiJSUzI1NiIsImtpZCI6IlhMRGZjbnNRemE4RGQyUXRMZG1MX3NXeG5TMzlQTnhnSHZkS1lGM25SODAifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJhcmdvIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImFyZ28tdG9rZW4tajd0a3MiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiYXJnbyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjBhNzI1N2JhLWZkZWQtNGI2OS05YWU2LTZhY2U0M2UxNjdlNiIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDphcmdvOmFyZ28ifQ.Gg7pctEsZC-2ZkFjHv-q21mOzBeuThocTMoNV2ZaLtOuxvXQOiVQhS8nq8nyPBiygJ3okOXKPhhrEH8Oe0kWuYtEXc88e1kX_MarQLCXLYSN53cdTLlgZQn01hHaHLO6KJubgU8mymNKj260GjDSf35a7wt8NgQIwm9ftqEwYuPXrm2yZEnhtbuNgfdpLIhw_DQxLXvwjTiny7vwR7ANpHfaynf2l0E12il3C7xeTP-lcPUm9BSFObO3icUbz67n0qsz3j8QWxRdH-jTzIr7tTvFP8SpdJbvMBmI4fgU01FR5CnWx296I9bzXjbuNefZGNu9ZuJ5RLiQDt5xmbTweQ"
from dflow.plugins import bohrium
bohrium.config["username"] = "xxx"
bohrium.config["password"] = "xxx"
bohrium.config["project_id"] = xxx
bohrium_context = BohriumContext(
        username="xxx",
        password="xxx",
        executor="mixed",
        extra={}
        )


def structure_optimization(structure_file: Path, pp: dict, trajectory_file: str) -> Path:

    from ase.io import read, write
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from ase.calculators.abacus import Abacus, AbacusProfile
    from ase.neb import NEB
    from ase.neb import NEBTools
    import matplotlib.pyplot as plt

    # Load structure
    structure = read(structure_file,format="abacus")

    # Set up the calculator
    abacus='/usr/local/bin/abacus'
    profile = AbacusProfile(argv=['mpirun','-n','8',abacus])
    kpts = [1,1,1]
    structure.calc = Abacus(profile=profile, ntype=2, ecutwfc=20, scf_nmax=50, smearing_method='gaussian', smearing_sigma=0.01, basis_type='pw', 
                mixing_type='pulay', mixing_beta='0.7', scf_thr=1e-8, out_chg=1, calculation='scf', force_thr=0.001, stress_thr=5,
                cal_force=1, cal_stress=1, out_stru=1, pp=pp, kpts=kpts)

    # Optimize structure
    mask = [atom.tag > 1 for atom in structure]
    structure.set_constraint(FixAtoms(mask=mask))

    opt = BFGS(structure, trajectory=trajectory_file)
    opt.run(fmax=0.05)

    return Path(trajectory_file)


def neb_calculation(initial_neb_structure: Path, 
                    final_neb_structure: Path,
                    num_images: int, 
                    pp: dict
                    ) -> Path:

    from ase.io import read, write
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from ase.calculators.abacus import Abacus, AbacusProfile
    from ase.neb import NEB
    from ase.neb import NEBTools
    import matplotlib.pyplot as plt


    # Load initial and final structures
    initial = read(initial_neb_structure)
    final = read(final_neb_structure)

    # Set up the calculator
    abacus='/usr/local/bin/abacus'
    profile = AbacusProfile(argv=['mpirun','-n','8',abacus])
    kpts = [1,1,1]
    calc = Abacus(profile=profile, ntype=2, ecutwfc=20, scf_nmax=50, smearing_method='gaussian', smearing_sigma=0.01, basis_type='pw', 
                mixing_type='pulay', mixing_beta='0.7', scf_thr=1e-8, out_chg=1, calculation='scf', force_thr=0.001, stress_thr=5,
                cal_force=1, cal_stress=1, out_stru=1, pp=pp, kpts=kpts)

    # Make a band consisting of 'num_images' number of images
    constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])
    images = [initial]
    for i in range(num_images):
        image = initial.copy()
        image.calc = Abacus(profile=profile, ntype=2, ecutwfc=20, scf_nmax=50, smearing_method='gaussian', smearing_sigma=0.01, basis_type='pw', 
                mixing_type='pulay', mixing_beta='0.7', scf_thr=1e-8, out_chg=1, calculation='scf', force_thr=0.001, stress_thr=5,
                cal_force=1, cal_stress=1, out_stru=1, pp=pp, kpts=kpts)
        image.set_constraint(constraint)
        images.append(image)
    images.append(final)

    # Optimize NEB path
    neb = NEB(images)
    neb.interpolate('idpp')
    opt = BFGS(neb, trajectory='neb.traj')
    opt.run(fmax=0.05,steps=10)

    return Path('neb.traj')


def neb_post_processing(neb_traj_file: Path, num_images: int ) -> float:

    from ase.io import read, write
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from ase.calculators.abacus import Abacus, AbacusProfile
    from ase.neb import NEB
    from ase.neb import NEBTools
    import matplotlib.pyplot as plt

    # Get the calculated barrier and the energy change of the reaction.
    images = read(f'{neb_traj_file}@-{num_images+2}:')
    nebtools = NEBTools(images)
    Ef, dE = nebtools.get_barrier()

    return Ef


class StructureOptimization(OP):
    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "opt_input_files": Artifact(Path),
            "pp": Parameter(dict)
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "opt_output_files": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        os.system("pip install git+https://gitlab.com/1041176461/ase-abacus.git")
        initial_structure = op_in["opt_input_files"]/"initial_stru"
        final_structure = op_in["opt_input_files"]/"final_stru"
        pp = op_in["pp"]

        os.chdir(op_in["opt_input_files"])
        initial_opt_traj = structure_optimization(initial_structure,pp,"initial.traj")
        final_opt_traj = structure_optimization(final_structure,pp,"final.traj")

        op_out = {
            "opt_output_files": [initial_opt_traj, final_opt_traj, op_in["opt_input_files"]/"Al_ONCV_PBE-1.0.upf", op_in["opt_input_files"]/"Au_ONCV_PBE-1.0.upf"]
        }
        return op_out

class NEBCalculation(OP):
    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "neb_input_files": Artifact(Path),
            "num_images": Parameter(int),
            "pp": Parameter(dict)
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "neb_traj": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        os.system("pip install git+https://gitlab.com/1041176461/ase-abacus.git")

        initial_neb_structure = op_in["neb_input_files"]/"initial.traj"
        final_neb_structure = op_in["neb_input_files"]/"final.traj"

        num_images = op_in["num_images"]
        pp = op_in["pp"]

        os.chdir(op_in["neb_input_files"])
        neb_traj = neb_calculation(initial_neb_structure,final_neb_structure, num_images, pp)

        op_out = {
            "neb_traj": neb_traj
        }
        return op_out

class NEBPostProcessing(OP):
    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign({
            "neb_traj": Artifact(Path),
            "num_images": Parameter(int)
        })

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign({
            "neb_image": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        os.system("pip install git+https://gitlab.com/1041176461/ase-abacus.git")

        neb_traj = op_in["neb_traj"]

        neb_image = neb_post_processing(neb_traj,num_images)

        op_out = {
            "neb_image": neb_image
        }
        return op_out


if __name__ == "__main__":

    artifact0 = upload_artifact(["./initial_stru","./final_stru","./Al_ONCV_PBE-1.0.upf","./Au_ONCV_PBE-1.0.upf"])

    pp = {'Al':'Al_ONCV_PBE-1.0.upf','Au':'Au_ONCV_PBE-1.0.upf'}

    # Create steps
    structure_optimization_step = Step(
        name="structure-optimization",
        template=PythonOPTemplate(StructureOptimization,image="registry.dp.tech/dptech/abacus:3.2.0"),
        executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c16_m64_cpu","projectId":11052, "jobType":"container"}),
        artifacts={"opt_input_files":artifact0},
        parameters={"pp": pp} 
    )

    neb_calculation_step = Step(
        name="neb-calculation",
        template=PythonOPTemplate(NEBCalculation,image="registry.dp.tech/dptech/abacus:3.2.0"),
        executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c16_m64_cpu","projectId":11052, "jobType":"container"}),
        artifacts={"neb_input_files": structure_optimization_step.outputs.artifacts["opt_output_files"]},
        parameters={"num_images": 3,"pp": pp}
    )

    neb_post_processing_step = Step(
        name="neb-post-processing",
        template=PythonOPTemplate(NEBPostProcessing,image="registry.dp.tech/dptech/abacus:3.2.0"),
        executor=BohriumExecutor(executor="bohrium_v2", extra={"scassType":"c8_m32_cpu","projectId":11052, "jobType":"container"}),
        artifacts={"neb_traj": neb_calculation_step.outputs.artifacts["neb_traj"]},
        parameters={"num_images": 3}
    )

    # Create workflow
    wf = Workflow(name="neb-workflow", context=bohrium_context, host="https://workflow.test.dp.tech")
    wf.add(structure_optimization_step)
    wf.add(neb_calculation_step)
    wf.add(neb_post_processing_step)

    # Submit workflow
    wf.submit()

    #download_artifact(structure_optimization_step.outputs.artifacts["initial_opt_traj"])
    #download_artifact(structure_optimization_step.outputs.artifacts["final_opt_traj"])
    #download_artifact(neb_calculation_step.outputs.artifacts["neb_traj"])
    #download_artifact(neb_post_processing_step.outputs.artifacts["neb_traj"])
