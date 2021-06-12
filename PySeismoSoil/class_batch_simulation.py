import os
import itertools
import multiprocessing as mp

from .class_simulation import (
    Linear_Simulation, Equiv_Linear_Simulation, Nonlinear_Simulation,
)

from . import helper_generic as hlp


class Batch_Simulation:
    """
    Run site response simulations in batch.

    Parameters
    ----------
    list_of_simulations : list
        A list of simulation objects. Valid simulation objects include objects
        from these classes: ``Linear_Simulation``, ``Equiv_Linear_Simulation``,
        and ``Nonlinear_Simulation``.

    Returns
    -------
    list_of_simulations : list
        Same as the input parameter `list_of_simulations`.
    n_simulations : int
        Number of simulations in the list.
    sim_type : {``Linear_Simulation``, ``Equiv_Linear_Simulation``, ``Nonlinear_Simulation``}
        The object type of the site response simulations.
    """
    def __init__(self, list_of_simulations):
        if not isinstance(list_of_simulations, list):
            raise TypeError('`list_of_simulations` should be a list.')
        if len(list_of_simulations) == 0:
            raise ValueError(
                '`list_of_simulations` should have at least one element.'
            )
        sim_0 = list_of_simulations[0]
        if not isinstance(
            sim_0,
            (Linear_Simulation, Equiv_Linear_Simulation, Nonlinear_Simulation),
        ):
            raise TypeError(
                'Elements of `list_of_simulations` should be of '
                'type `Linear_Simulation`, `Equiv_Linear_Simulation`, '
                'or `Nonlinear_Simulation`.'
            )
        if not all(isinstance(i, type(sim_0)) for i in list_of_simulations):
            raise TypeError(
                'All the elements of `list_of_simulations` should be of the same type.'
            )
        n_simulations = len(list_of_simulations)

        self.list_of_simulations = list_of_simulations
        self.n_simulations = n_simulations
        self.sim_type = type(sim_0)

    def run(self, parallel=False, n_cores=1, base_output_dir=None, options={}):
        """
        Run simulations in batch.

        Parameters
        ----------
        parallel : bool
            Whether to use multiple CPU cores to run simulations.
        n_core : int or ``None``
            Number of CPU cores to be used. If ``None``, all CPU cores will be
            used.
        base_output_dir : str
            The parent directory for saving the output files/figures of the
            current batch.
        options : dict
            Options to be passed to the ``run()`` methods of the relevant
            simulation classes (linear, equivalent linear, or nonlinear). Check
            out the API documentation of the ``run()`` methods here:
            https://pyseismosoil.readthedocs.io/en/stable/api_docs/class_simulation.html

        Returns
        -------
        sim_results : list<Simulation_Result>
            Simulation results corresponding to each simulation object.
        """
        N = self.n_simulations
        n_digits = len(str(N))

        if base_output_dir is None:
            current_time = hlp.get_current_time(for_filename=True)
            base_output_dir = os.path.join('./', 'batch_sim_%s' % current_time)

        other_params = [n_digits, base_output_dir, options]

        if not parallel:
            sim_results = []
            for i in range(self.n_simulations):
                sim_results.append(self._run_single_sim([i, other_params]))
            # END FOR
        else:
            # Because no outputs can be printed to stdout in the parellel pool
            if options.get('verbose', True):  # default value is `True`
                print('Parallel computing in progress...', end=' ')
            p = mp.Pool(n_cores)
            sim_results = p.map(
                self._run_single_sim,
                itertools.product(range(N), [other_params]),
            )
            if options.get('verbose', True):
                print('done.')

            # Because no figures can be plotted in the parallel pool:
            if options.get('show_fig', False):
                for sim_result in sim_results:
                    sim_result.plot(save_fig=options.get('save_fig', False))
                # END FOR
            # END IF
        # END IF

        return sim_results

    def _run_single_sim(self, all_params):
        """
        Helper function to run a single simulation.

        Parameters
        ----------
        all_params : list
            All the parameters needed for running the simulation. It should
            have the following structure:
                i, n_digits, base_output_dir, options
            where:
                - ``i`` is the index of the current simulation in the batch.
                - ``n_digits`` is the number of digits of the length of the
                  batch. (For example, if there are 125 simulations, then
                  ``n_digits`` should be 3.)
                - ``base_output_dir``: same as in the ``run()`` method
                - ``options``: same as in the ``run()`` method

        Returns
        -------
        sim_result : PySeismoSoil.class_simulation_result.Simulation_Result
            Simulation result of a single simulation object.
        """
        i, other_params = all_params  # unpack
        n_digits, base_output_dir, options = other_params  # unpack
        output_dir = os.path.join(base_output_dir, str(i).rjust(n_digits, '0'))
        if self.sim_type == Nonlinear_Simulation:
            options.update({'sim_dir': output_dir})
        else:  # linear or equivalent linear
            options.update({'output_dir': output_dir})

        sim_obj = self.list_of_simulations[i]
        sim_result = sim_obj.run(**options)
        return sim_result
