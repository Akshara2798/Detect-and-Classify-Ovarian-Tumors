import numpy as np
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf
class HybridGGO_DBO(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                  amsgrad=False, weight_decay=None, clipnorm=None, clipvalue=None,
                  global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                  ema_overwrite_frequency=None, jit_compile=True, name="HybridGGO_DBO", **kwargs):
        super().__init__(name=name, weight_decay=weight_decay, clipnorm=clipnorm,
                          clipvalue=clipvalue, global_clipnorm=global_clipnorm, use_ema=use_ema,
                          ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency,
                          jit_compile=jit_compile, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.archive = [] 
   
    def objective_function(self,position):
        """ Define your objective function here. """
        return np.sum(position**2)  

    def initialize_agents(self,num_agents, bounds):
        """ Initializes the agents within the given bounds. """
        global agents, fitness_values, best_positions, best_fitness
        agents = np.random.uniform(bounds[0], bounds[1], (num_agents, len(bounds[0])))
        fitness_values = np.zeros(num_agents)
        best_positions = np.copy(agents)
        best_fitness = np.copy(fitness_values)

    def ggo_agent(self,agent):
        """ GGO logic for agent update (random movement). """
        return agent + np.random.uniform(-1, 1, (1,1))

    def dbo_agent(self,position, agent_type):
        """ DBO logic for different types of agents. """
        if agent_type == 'ball_rolling':
            return self.ball_rolling_dung_beetle(position)
        elif agent_type == 'brood_ball':
            return self.brood_ball(position)
        elif agent_type == 'small_dung_beetle':
            return self.small_dung_beetle(position)
        elif agent_type == 'thief':
            return self.thief(position)

    def ball_rolling_dung_beetle(self,position):
        """ Update logic for ball-rolling dung beetle. """
        return position + np.random.uniform(-0.5, 0.5, position.shape)

    def brood_ball(self,position):
        """ Update logic for brood ball dung beetle. """
        return position + np.random.uniform(-0.3, 0.3, position.shape)

    def small_dung_beetle(self,position):
        """ Update logic for small dung beetle. """
        return position + np.random.uniform(-0.2, 0.2, (1,1))

    def thief(self,position):
        """ Update logic for thief beetle. """
        return position + np.random.uniform(-0.4, 0.4, (1,1))

    def calculate_objective_values(self,agents):
        """ Calculate the objective values for all agents. """
        return np.array([[self.objective_function(agent), np.prod(agent)] for agent in agents])

    def find_non_dominated_solutions(self,objective_values):
        """ Find non-dominated solutions using Pareto dominance. """
        non_dominated = []
        for i, val1 in enumerate(objective_values):
            dominated = False
            for j, val2 in enumerate(objective_values):
                if all(val2 <= val1) and any(val2 < val1):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
        return [objective_values[i] for i in non_dominated]
    
    
    
    
    

    def update_archive(self, new_solutions):
        """ Update the archive with new non-dominated solutions. """
        self.archive.extend(new_solutions)  # Use self.archive
        self.archive = list({tuple(sol): sol for sol in self.archive}.values())  # Remove duplicates
        if len(self.archive) > 100:  # Keep the archive size manageable
            self.archive = self.archive[:100]

    def hybrid_ggo_dbo(self, gradient):
        # Define the agents and initialize them
        bounds = [np.array([-10, -10]), np.array([10, 10])]
        num_agents = 30
        num_objectives = 2
        max_iterations = 100
        
        # Initialize agents and fitness values
        self.initialize_agents(num_agents, bounds)

        # Initialize fitness values and best positions
        fitness_values = np.full(num_agents, np.inf)  # Initialize with infinity for minimization
        best_positions = np.copy(agents)
        best_fitness = np.full(num_agents, np.inf)  # Initialize best fitness values

        objective_values = self.calculate_objective_values(agents)
        self.archive = self.find_non_dominated_solutions(objective_values)  # Use self.archive

        t = 0
        while t < max_iterations:
            for i in range(num_agents):
                if np.random.rand() < 0.5:
                    agents[i] = self.ggo_agent(agents[i])  # GGO update
                else:
                    agent_type = np.random.choice(['ball_rolling', 'brood_ball', 'small_dung_beetle', 'thief'])
                    agents[i] = self.dbo_agent(agents[i], agent_type)  # DBO update

                agents[i] = np.clip(agents[i], bounds[0], bounds[1])  # Keep within bounds

                # Update fitness
                fitness_values[i] = self.objective_function(agents[i])

                if fitness_values[i] < best_fitness[i]:
                    best_fitness[i] = fitness_values[i]
                    best_positions[i] = agents[i]

            objective_values = self.calculate_objective_values(agents)
            new_solutions = self.find_non_dominated_solutions(objective_values)
            self.update_archive(new_solutions)  # This is now correct

            t += 1

        best_index = np.argmin(best_fitness)
            

        
        
    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        self.hybrid_ggo_dbo(gradient)
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        if isinstance(gradient, tf.IndexedSlices):
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config

 
 



