# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import random
import contest.util as util
import pickle
import os
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveQLearningAgent', second='DefensiveReflexedAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
class OffensiveQLearningAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1, num_training=0, epsilon=0.0, alpha=0.6, gamma=0.7):
        super().__init__(index, time_for_computing)
        self.q_values = util.Counter()  # Almacena los valores Q
        self.epsilon = epsilon  # Probabilidad de explorar acciones aleatorias
        self.alpha = alpha  # Tasa de aprendizaje
        self.discount = gamma  # Factor de descuento para los futuros valores Q
        self.num_training = num_training  # Número de episodios de entrenamiento
        self.episodes_so_far = 0  # Contador de episodios jugados

        self.q_values_file = "./offensive_q_values.pkl"  # Ruta para guardar los valores Q

    def register_initial_state(self, game_state):
        # Inicializa el agente al comienzo del juego.
        super().register_initial_state(game_state)
        self.load_q_values()  # Carga los valores Q desde un archivo
        self.start = game_state.get_agent_position(self.index)  # Obtiene la posición inicial

    def choose_action(self, game_state):
        # Elegir una acción.
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return None

        # Exploración
        if util.flip_coin(self.epsilon):  # Exploración aleatoria
            chosen_action = random.choice(legal_actions)
        else:
            # Elegir la mejor acción basada en los valores Q
            chosen_action = self.compute_action_from_q_values(game_state)

        # Obtener el siguiente estado y calcular la recompensa
        successor_state = self.get_successor(game_state, chosen_action)
        reward = self.get_reward(successor_state)

        # Actualizar los valores Q con la transición observada
        self.update(game_state, chosen_action, successor_state, reward)

        return chosen_action

    def get_successor(self, game_state, action):
        # Devuelve el estado sucesor después de realizar la acción dada.
        successor = game_state.generate_successor(self.index, action)
        return successor

    def get_features(self, game_state, action):
        # Extraer las características para un par estado-acción dado.
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()

        # Característica: Distancia a la comida más cercana
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Característica: Proximidad a fantasmas
        ghost_indices = [i for i in self.get_opponents(game_state) if not game_state.get_agent_state(i).is_pacman]
        ghost_states = [game_state.get_agent_state(i) for i in ghost_indices]
        ghost_positions = [ghost.get_position() for ghost in ghost_states if ghost.get_position()]

        if ghost_positions:
            ghost_distances = [self.get_maze_distance(my_pos, ghost_pos) for ghost_pos in ghost_positions]
            min_distance = min(ghost_distances)

            # Verificar si los fantasmas están asustados
            scared_ghosts = [ghost for ghost in ghost_states if ghost.scared_timer > 0]
            if scared_ghosts:
                # Si los fantasmas están asustados, fomentar moverse hacia ellos
                features['ghost_proximity'] = 1 / min_distance if min_distance > 0 else 10
            else:
                # Si los fantasmas no están asustados, evitarlos
                if min_distance == 0:  # Fantasma en la misma posición
                    features['ghost_proximity'] = -float('inf')  # Penalización máxima
                elif min_distance < 3:
                    features['ghost_proximity'] = -4 / min_distance  # Penalización ajustada
                else:
                    features['ghost_proximity'] = 0  # Distancia segura
        else:
            features['ghost_proximity'] = 0  # No hay fantasmas cerca

        # Característica: Distancia a casa si lleva suficiente comida
        agent_state = game_state.get_agent_state(self.index)
        if agent_state.num_carrying >= 1:  # Activar esta característica si lleva 1
            home_distance = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = home_distance
        else:
            features['distance_to_home'] = 0

        # Característica: Cantidad de comida cercana
        features['successor_score'] = -len(food_list)

        return features

    def get_weights(self):
        # Define los pesos para las características en la evaluación estado-acción.
        return {
            'distance_to_food': -1,        # Incentivar acercarse a la comida
            'ghost_proximity': 300,       # Penalizar fuertemente estar cerca de fantasmas
            'distance_to_home': -10,      # Incentivar regresar a casa si lleva comida suficiente
            'successor_score': 100        # Recompensar áreas con más comida
        }

    def compute_q_value(self, game_state, action):
        # Computar el valor Q a partir de las características y los pesos
        features = self.get_features(game_state, action)
        weights = self.get_weights()
        return sum(features[feature] * weights[feature] for feature in features)

    def compute_action_from_q_values(self, game_state):
        # Elegir la acción con el mejor valor Q
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return None

        max_q_value = float('-inf')
        best_actions = []
        for action in legal_actions:
            q_value = self.compute_q_value(game_state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                best_actions = [action]
            elif q_value == max_q_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def update(self, state, action, next_state, reward):
        # Actualizar los valores Q después de realizar la acción con las recompensas dadas
        current_q_value = self.compute_q_value(state, action)
        next_q_value = self.compute_value_from_q_values(next_state)
        sample = reward + self.discount * next_q_value
        self.q_values[(state, action)] = current_q_value + self.alpha * (sample - current_q_value)

    def compute_value_from_q_values(self, game_state):
        # Calcular el valor máximo de los valores Q
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return 0.0
        return max(self.compute_q_value(game_state, action) for action in legal_actions)

    def get_reward(self, game_state):
        # Definir las recompensas y penalizaciones para el agente
        reward = 0
        agent_state = game_state.get_agent_state(self.index)
        agent_position = agent_state.get_position()

        if agent_state.get_position() is None:
            reward -= 1000  # Penalización alta si es atrapado

        if agent_state.num_carrying >= 6:  # Incentivar regresar a casa si lleva comida suficiente
            home_distance = self.get_maze_distance(agent_position, self.start)
            reward += max(0, 10 - home_distance * 0.1)  # Recompensa por acercarse a casa

        reward -= 0.1  # Penalización por cada paso (eficiencia)

        return reward

    def observe_transition(self, state, action, next_state, reward):
        # Registra la transición y actualiza los valores Q
        self.update(state, action, next_state, reward)

    def final(self, game_state):
        # Llamado al final del juego para guardar los valores Q
        self.episodes_so_far += 1
        if self.episodes_so_far == self.num_training:
            self.epsilon = 0.0
            self.alpha = 0.0
        #self.save_q_values() no mas save q_values ya que ocupan mucho espacio en el archivo

    def save_q_values(self):
        # Guardar los valores Q en un archivo.
        try:
            with open(self.q_values_file, "wb") as f:
                pickle.dump(self.q_values, f)
            print(f"Valores Q guardados en {self.q_values_file}")
        except Exception as e:
            print(f"No se pudieron guardar los valores Q: {e}")

    def load_q_values(self):
        # Cargar los valores Q desde un archivo.
        try:
            with open(self.q_values_file, "rb") as f:
                self.q_values = pickle.load(f)
            print(f"Valores Q cargados desde {self.q_values_file}")
        except FileNotFoundError:
            print("No se encontró archivo de valores Q. Comenzando desde cero.")

class DefensiveReflexedAgent(CaptureAgent):
    # Agente defensivo reflexivo que patrulla su lado del mapa y evita cruzar al lado enemigo.
    
    def register_initial_state(self, game_state):
        # Inicializa el agente al comienzo del juego.
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)

        # Determina la línea central (línea media) del mapa y restringe el cruce.
        layout_width = game_state.data.layout.width
        self.mid_x = layout_width // 2
        if not game_state.is_on_red_team(self.index):
            self.mid_x -= 1  # Ajuste para el equipo azul.

    def choose_action(self, game_state):
        # Elige la mejor acción basada en la evaluación de características.
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP

        # Filtra las acciones que cruzan al lado enemigo.
        legal_actions = [action for action in legal_actions if not self.crosses_to_enemy(game_state, action)]

        if not legal_actions:
            return Directions.STOP

        # Evalúa cada acción basada en características y selecciona la mejor.
        scores = [(self.evaluate_action(game_state, action), action) for action in legal_actions]
        return max(scores, key=lambda x: x[0])[1]

    def crosses_to_enemy(self, game_state, action):
        # Verifica si la acción haría que el agente cruce al lado enemigo.
        successor = self.get_successor(game_state, action)
        x, _ = successor.get_agent_position(self.index)
        if game_state.is_on_red_team(self.index) and x > self.mid_x:
            return True
        elif not game_state.is_on_red_team(self.index) and x < self.mid_x:
            return True
        return False

    def evaluate_action(self, game_state, action):
        # Evalúa una acción calculando una suma ponderada de sus características.
        features = self.get_features(game_state, action)
        weights = self.get_weights()
        return sum(features[feature] * weights.get(feature, 0) for feature in features)

    def get_features(self, game_state, action):
        # Extrae las características relevantes defensivas para la evaluación.
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Característica 1: Estatus defensivo (no un Pacman)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Característica 2: Número de invasores visibles
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Característica 3: Distancia al invasor más cercano
        if invaders:
            distances = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(distances)
        else:
            features['invader_distance'] = 0

        # Característica 4: Distancia a la comida más cercana que se está defendiendo
        food_list = self.get_food_you_are_defending(successor).as_list()
        if food_list:
            distances = [self.get_maze_distance(my_pos, food) for food in food_list]
            features['distance_to_food'] = min(distances)
        else:
            features['distance_to_food'] = 0

        # Característica 5: Penaliza detenerse
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        # Característica 6: Penaliza invertir la dirección
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def get_weights(self):
        # Asigna pesos a cada característica para la evaluación.
        return {
            'on_defense': 100,         # Gran incentivo para mantenerse en defensa
            'num_invaders': -1000,     # Gran penalización por invasores en nuestro lado
            'invader_distance': -10,   # Incentivo para acercarse a los invasores
            'distance_to_food': -1,    # Incentivo para patrullar cerca de la comida
            'stop': -100,              # Penaliza detenerse
            'reverse': -2              # Penaliza invertir la dirección
        }

    def get_successor(self, game_state, action):
        # Genera el siguiente estado del juego después de tomar una acción.
        return game_state.generate_successor(self.index, action)