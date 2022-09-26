"""
Conventional BP decoder

Brief: Conventional BP decoder

Copyright (c) 2022 Orange

Author: Guillaume Larue <guillaume.larue@orange.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""

import numpy as np
import tensorflow as tf


class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = []
        self.node_value = 0

    def __repr__(self):
        return "{classname}({name}, [{neighbors}])".format(
            classname=type(self).__name__,
            name=self.name,
            neighbors=", ".join([n.name for n in self.neighbors]),
        )

    def is_valid_neighbor(self, neighbor):
        raise NotImplemented()

    def add_neighbor(self, neighbor):
        assert self.is_valid_neighbor(neighbor)
        self.neighbors.append(neighbor)


class Variable(Node):
    def __init__(self, name, channel_value):
        self.name = name
        self.neighbors = []
        self.node_value = 0
        self.channel_value = channel_value

    def is_valid_neighbor(self, check):
        return isinstance(check, Check)

    def set_channel_value(self, channel_value):
        self.channel_value = channel_value


class Check(Node):
    def is_valid_neighbor(self, variable):
        return isinstance(variable, Variable)


class Messages:
    def __init__(self):
        self._messages = {}

    def init_message(self, node_1, node_2, message_value=0):
        message_name = (node_1.name, node_2.name)
        self._messages[message_name] = message_value

    def variable_to_check_message(self, variable, check):
        raise NotImplemented()

    def check_to_variable_message(self, check, variable):
        raise NotImplemented()

    def get_variable_to_check_message(self, variable, check):
        message_name = (variable.name, check.name)
        """
        if message_name not in self._messages:
            self._messages[message_name] = 0#np.inf
        """
        return self._messages[message_name]

    def get_check_to_variable_message(self, check, variable):
        message_name = (check.name, variable.name)
        """
        if message_name not in self._messages:
            self._messages[message_name] = 0
        """
        return self._messages[message_name]


class SumProduct(Messages):
    def variable_to_check_message(self, variable, check):
        variable_to_check_message_name = (variable.name, check.name)
        # incoming_messages = []
        message_value = variable.channel_value
        for neighbor_check in variable.neighbors:
            if neighbor_check.name != check.name:
                # incoming_messages.append(self.get_check_to_variable_message(neighbor_check,variable))
                message_value += self.get_check_to_variable_message(
                    neighbor_check, variable
                )
        # message_value = np.sum(incoming_messages)+variable.channel_value
        # variable_value = message_value + self.get_check_to_variable_message(check,variable)
        # variable.node_value = variable_value
        self._messages[variable_to_check_message_name] = message_value
        return message_value

    def check_to_variable_message(self, check, variable):
        check_to_variable_message_name = (check.name, variable.name)
        # incoming_messages = []
        product = 1
        for neighbor_variable in check.neighbors:
            if neighbor_variable.name != variable.name:
                # incoming_messages.append(self.get_variable_to_check_message(neighbor_variable, check))
                product = product * np.tanh(
                    self.get_variable_to_check_message(neighbor_variable, check) / 2
                )
                # product = product * np.clip(np.tanh(self.get_variable_to_check_message(neighbor_variable, check)/2),-0.99,+0.99)
        # incoming_messages = np.array(incoming_messages)
        # product = np.prod(np.tanh(incoming_messages/2))
        #!message_value = 2 * np.arctanh(product)
        clip_value = 18.71497388
        message_value = 2 * np.clip(np.arctanh(product), -clip_value, +clip_value)  #!
        # check_value = 2*np.arctanh(product*np.tanh(self.get_variable_to_check_message(variable, check)/2))
        # check.node_value = check_value
        self._messages[check_to_variable_message_name] = message_value
        return message_value


class MinSum(Messages):
    def variable_to_check_message(self, variable, check):
        variable_to_check_message_name = (variable.name, check.name)
        incoming_messages = []
        for neighbor_check in variable.neighbors:
            if neighbor_check.name != check.name:
                incoming_messages.append(
                    self.get_check_to_variable_message(neighbor_check, variable)
                )

        message_value = np.sum(incoming_messages) + variable.channel_value
        # variable_value = message_value + self.get_check_to_variable_message(check,variable)
        # variable.node_value = variable_value
        self._messages[variable_to_check_message_name] = message_value
        return message_value

    def check_to_variable_message(self, check, variable):
        check_to_variable_message_name = (check.name, variable.name)
        # incoming_messages = []
        message_value = 1
        min_message = np.inf
        for neighbor_variable in check.neighbors:
            if neighbor_variable.name != variable.name:
                temp_message = self.get_variable_to_check_message(
                    neighbor_variable, check
                )
                abs_temp_message = np.abs(temp_message)
                if abs_temp_message < min_message:
                    min_message = abs_temp_message
                message_value = message_value * np.sign(temp_message)
                # incoming_messages.append(self.get_variable_to_check_message(neighbor_variable, check))
        # incoming_messages = np.array(incoming_messages)
        # message_value = np.prod(np.sign(incoming_messages))*np.min(np.abs(a))
        message_value = message_value * min_message
        # check.node_value = check_value
        self._messages[check_to_variable_message_name] = message_value
        return message_value


class FactorGraph:
    # algorithm can be set either to "SumProduct()" or "MinSum()" depending on the algorithm to be used
    def __init__(self, name, H, systematic_bits, algorithm=SumProduct()):
        self.name = name
        self.H = H
        self.systematic_bits = np.array(systematic_bits)
        self.number_check_nodes = H.shape[0]
        self.number_variable_nodes = H.shape[1]
        self.check_nodes = []
        self.variable_nodes = []
        self.messages = algorithm

        # Create check nodes:
        for c in range(self.number_check_nodes):
            self.check_nodes.append(Check("c" + str(c)))

        # Create variable nodes:
        for v in range(self.number_variable_nodes):
            self.variable_nodes.append(Variable(name="v" + str(v), channel_value=0))

        # Create graph:
        for c in range(self.number_check_nodes):
            for v in range(self.number_variable_nodes):
                if H[c][v] == 1:
                    self.check_nodes[c].add_neighbor(self.variable_nodes[v])
                    self.variable_nodes[v].add_neighbor(self.check_nodes[c])

        # Init variable to check nodes messages:
        for variable in self.variable_nodes:
            for neighbor_check in variable.neighbors:
                self.messages.init_message(variable, neighbor_check, message_value=0)

        # Init variable to check nodes messages:
        for check in self.check_nodes:
            for neighbor_variable in check.neighbors:
                self.messages.init_message(check, neighbor_variable, message_value=0)

    def _reset_graph(self):
        self.__init__(self.name, self.H, self.systematic_bits, self.messages)

    def __repr__(self):
        return "{classname}({name}, [{check_nodes}],[{variable_nodes}])".format(
            classname=type(self).__name__,
            name=self.name,
            check_nodes=", ".join([str(check) for check in self.check_nodes]),
            variable_nodes=", ".join(
                [str(variable) for variable in self.variable_nodes]
            ),
        )

        for c in range(self.number_check_nodes):
            print(self.check_nodes[c])

        for v in range(self.number_variable_nodes):
            print(self.variable_nodes[v])

    def _variables_to_checks_iteration(self):
        # Iteration "SUM"
        for variable_node in self.variable_nodes:
            for neighbor_check_node in variable_node.neighbors:
                self.messages.variable_to_check_message(
                    variable_node, neighbor_check_node
                )

    def _checks_to_variables_iteration(self):
        # Iteration "PRODUCT"
        for check_node in self.check_nodes:
            for neighbor_variable_node in check_node.neighbors:
                self.messages.check_to_variable_message(
                    check_node, neighbor_variable_node
                )

    def _variables_values(self):
        for variable_node in self.variable_nodes:
            messages_sum = variable_node.channel_value
            for neighbor_check_node in variable_node.neighbors:
                messages_sum += self.messages.get_check_to_variable_message(
                    neighbor_check_node, variable_node
                )
            variable_node.node_value = messages_sum

    def _set_channel_values(self, channel_values):
        assert self.number_variable_nodes == len(channel_values)
        self._reset_graph()
        for v in range(self.number_variable_nodes):
            self.variable_nodes[v].set_channel_value(channel_values[v])
            self.variable_nodes[v].node_value = channel_values[v]

    def _iterate(self, n_iteration_max, n_iteration_min):
        success = False

        for iteration in range(n_iteration_max):
            self._variables_to_checks_iteration()
            self._checks_to_variables_iteration()
            self._variables_values()
            coded_word = [
                self._decode_bit(variable.node_value)
                for variable in self.variable_nodes
            ]

            # Exit chart: If coded word exists, i.e. C.H^T = 0
            if (
                np.sum(np.matmul(coded_word, np.transpose(self.H)) % 2) == 0
                and (iteration + 1) >= n_iteration_min
            ):
                success = True
                return (self._decode_word(coded_word), success)

        return (self._decode_word(coded_word), success)

    def _decode_bit(self, LLR):

        return (
            -np.sign(LLR) + 1
        ) / 2  #! np.array((-np.sign(LLR) + 1) / 2, dtype=np.uint8)
        """
        return LLR
        """

    def _decode_word(self, coded_words):
        return np.array(coded_words)[self.systematic_bits]

    def decode(self, LLRs, max_iteration, min_iteration=0):
        decoded_words = []
        for LLR in LLRs:
            self._reset_graph()
            # Exit chart: If received coded word exists, i.e. C.H^T = 0
            if (
                np.sum(np.matmul(self._decode_bit(LLR), np.transpose(self.H)) % 2) == 0
                and min_iteration == 0
            ):
                decoded_words.append(self._decode_word(self._decode_bit(LLR)))
            else:
                self._set_channel_values(LLR)
                decoded_words.append(self._iterate(max_iteration, min_iteration)[0])
            #!self._set_channel_values(LLR)
            #!decoded_words.append(self._iterate(max_iteration, min_iteration)[0])
        return decoded_words
