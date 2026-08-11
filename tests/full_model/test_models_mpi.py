"""
This module contains tests that run mozaik models and compare their output to a
saved reference.
"""

import numpy as np
import os
import sys
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore
from mozaik.tools.distribution_parametrization import PyNNDistribution
from parameters import ParameterSet
from .test_models import PROJECT_ROOT, TestModel

import pytest
import mozaik


@pytest.mark.model
@pytest.mark.mpi
@pytest.mark.stepcurrentmodule
class TestLSV1MTinyOptoMPI:
    @pytest.fixture(scope="class")
    def lsv1m_tiny_opto_mpi(self):
        model_directory = os.path.join(
            PROJECT_ROOT, "tests", "full_model", "models", "LSV1M_tiny_opto"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (PROJECT_ROOT, env.get("PYTHONPATH")))
        )
        data_stores = {}
        for num_processes in (1, 2, 4):
            threads_per_process = 4 // num_processes
            run_name = "pytest_mpi%d" % num_processes
            result_path = os.path.join(model_directory, "LSV1M_%s_____" % run_name)
            data_stores[num_processes] = TestModel.run_model_and_load(
                [
                    "mpirun",
                    "--oversubscribe",
                    "-np",
                    str(num_processes),
                    sys.executable,
                    "run.py",
                    "nest",
                    str(threads_per_process),
                    "param/defaults_mpi",
                    run_name,
                ],
                result_path,
                cwd=model_directory,
                env=env,
                shell=False,
            )

        return data_stores

    @pytest.fixture(scope="class")
    def lsv1m_tiny_opto_mpi_reference(self):
        reference_path = os.path.join(
            os.path.dirname(__file__), "reference_data", "LSV1M_tiny_opto_mpi"
        )
        return TestModel.load_datastore(reference_path)

    @pytest.mark.parametrize("num_processes", (1, 2, 4))
    def test_spikes(
        self,
        lsv1m_tiny_opto_mpi,
        lsv1m_tiny_opto_mpi_reference,
        num_processes,
    ):
        TestModel().check_spikes(
            lsv1m_tiny_opto_mpi[num_processes],
            lsv1m_tiny_opto_mpi_reference,
            sheet_name="V1_Exc_L2/3",
        )

    @pytest.mark.parametrize("num_processes", (1, 2, 4))
    def test_voltages(
        self,
        lsv1m_tiny_opto_mpi,
        lsv1m_tiny_opto_mpi_reference,
        num_processes,
    ):
        TestModel().check_voltages(
            lsv1m_tiny_opto_mpi[num_processes],
            lsv1m_tiny_opto_mpi_reference,
            sheet_name="V1_Exc_L2/3",
            max_neurons=25,
        )


class TestLSV1MTinyMPI(TestModel):
    """
    Class that runs the a tiny version of the LSV1M model on construction from the mozaik-models
    repository and runs it with MPI. Its testing methods compare the membrane potentials of a
    few neurons and the spike times of all neurons to a saved reference.
    """

    model_run_command = (
        "cd tests/full_model/models/LSV1M_tiny && "
        f"mpirun -np 2 {sys.executable} run.py nest 1 "
        "param/defaults 'pytest_mpi2' && cd ../../../.."
    )
    result_path = "tests/full_model/models/LSV1M_tiny/LSV1M_pytest_mpi2_____"
    ref_path = "tests/full_model/reference_data/LSV1M_tiny_mpi"
    model_run_uses_posix_spawn = True

    ds = None  # Model run datastore
    ds_ref = None  # Reference datastore

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.parametrize(
        "sheet_name", ["V1_Exc_L4", "V1_Inh_L4", "V1_Exc_L2/3", "V1_Inh_L2/3"]
    )
    def test_spikes(self, sheet_name):
        self.check_spikes(self.ds, self.ds_ref, sheet_name)

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.parametrize(
        "sheet_name", ["V1_Exc_L4", "V1_Inh_L4", "V1_Exc_L2/3", "V1_Inh_L2/3"]
    )
    def test_voltages(self, sheet_name):
        self.check_voltages(self.ds, self.ds_ref, sheet_name, max_neurons=25)

    @pytest.mark.model
    @pytest.mark.mpi
    def test_mozaik_rng_mpi2(self):
        rngs_state = self.ds.block.annotations["simulation_log"]["rngs_state"]
        print(rngs_state)
        assert len(set(rngs_state)) == 1


class TestModelExplosionMonitoringMPI(TestModel):
    """
    Tests whether the explosion monitoring works as expected
    """

    model_run_command = (
        "cd tests/full_model/models/LSV1M_tiny && "
        f"mpirun -np 2 {sys.executable} run.py nest 1 "
        "param/defaults_explosion 'pytest_mpi_explosion' && cd ../../../.."
    )
    result_path = "tests/full_model/models/LSV1M_tiny/LSV1M_pytest_mpi_explosion_____"
    ref_path = "tests/full_model/reference_data/LSV1M_tiny_mpi"
    model_run_uses_posix_spawn = True

    ds = None  # Model run datastore
    ds_ref = None  # Reference datastore

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.mpi_explosion
    def test_explosion(self):
        assert self.ds.block.annotations["simulation_log"]["explosion_detected"]

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.mpi_explosion
    def test_fr_above_threshold(self):
        sheet_monitored = eval(self.ds.get_model_parameters())["explosion_monitoring"][
            "sheet_name"
        ]
        threshold = eval(self.ds.get_model_parameters())["explosion_monitoring"][
            "threshold"
        ]
        last_seg = param_filter_query(
            self.ds, sheet_name=sheet_monitored
        ).get_segments()[-1]
        assert (
            numpy.mean(
                [
                    len(st) / (st.t_stop - st.t_start) * 1000
                    for st in last_seg.spiketrains
                ]
            )
            > threshold
        )


class TestLSV1MTinyMPI7(TestModel):
    """
    Class that runs the a tiny version of the LSV1M model on construction from the mozaik-models
    repository and runs it with MPI using 7 processes. Its testing methods compare the membrane
    potentials of a few neurons and the spike times of all neurons to a saved reference.
    """

    model_run_command = (
        "cd tests/full_model/models/LSV1M_tiny && "
        f"mpirun --oversubscribe -np 7 {sys.executable} run.py nest 1 "
        "param/defaults 'pytest_mpi7' && cd ../../../.."
    )
    result_path = "tests/full_model/models/LSV1M_tiny/LSV1M_pytest_mpi7_____"
    ref_path = "tests/full_model/reference_data/LSV1M_tiny_mpi"
    model_run_uses_posix_spawn = True

    ds = None  # Model run datastore
    ds_ref = None  # Reference datastore

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.not_github
    def test_mozaik_rng_mpi7(self):
        rngs_state = self.ds.block.annotations["simulation_log"]["rngs_state"]
        print(rngs_state)
        assert len(set(rngs_state)) == 1
