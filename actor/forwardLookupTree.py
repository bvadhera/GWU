import uuid
import gc

def sanitize_id(id):
    return id.strip().replace(" ", "")

(_ADD, _DELETE, _INSERT) = range(3)
(_ROOT, _DEPTH, _WIDTH) = range(3)

class Node:

    def __init__(self, name, identifier=None, expanded=True):
        self.__identifier = (str(uuid.uuid1()) if identifier is None else
                sanitize_id(str(identifier)))
        self.name = name
        self.expanded = expanded
        self.__utility = 0.0 

    @property
    def identifier(self):
        return self.__identifier


    @property
    def utility(self):
        return self.__utility
    
    @utility.setter
    def utility(self, value):
        if value is not None:
            self.__utility = value
 

    @property
    def depth(self):
        return self.__depth
    
    @depth.setter
    def depth(self, value):
        if value is not None:
            self.__depth = value

    @property
    def reward(self):
        return self.__reward
    
    @reward.setter
    def reward(self, value):
        if value is not None:
            self.__reward = value

    @property
    def updated_reward(self):
        return self.__updated_reward
    
    @updated_reward.setter
    def updated_reward(self, value):
        if value is not None:
            self.__updated_reward = value

    @property
    def state(self):
        return self.__state

        
    
    @state.setter
    def state(self, value):
        if value is not None:
            self.__state = value
            
    @property
    def max_state(self):
        return self.__max_state
    
    @max_state.setter
    def max_state(self, value):
        if value is not None:
            self.__max_state = value


class Tree:

    def __init__(self):
        self.nodes = []
    
    def get_nodes(self):
        return self.nodes

    def set_nodes(self, nodeList):
        self.nodes = nodeList
       

    def get_index(self, position):
        for index, node in enumerate(self.nodes):
            if node.identifier == position:
                break
        return index
    
    def get_node(self, position):
        return_node = None
        for index, node in enumerate(self.nodes):
            if node.identifier == position:
                return_node = node
                break
        return return_node
                    
    def create_node(self, name, reward,updated_reward,state, max_state, depth = None, identifier=None, utility=None):
        node = Node(name, identifier)
        self.nodes.append(node)
        node.utility = utility
        node.reward = reward
        node.updated_reward = updated_reward
        node.state = state
        node.max_state = max_state
        node.depth = depth
        return node

    def delete_node(self,position):
        for index, node in enumerate(self.nodes):
            if node.identifier == position:
                self.nodes.pop(index)
                del node 
                break

 
    def __getitem__(self, key):
        return self.nodes[self.get_index(key)]

    def __setitem__(self, key, item):
        self.nodes[self.get_index(key)] = item

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, identifier):
        return [node.identifier for node in self.nodes if node.identifier is identifier]


if __name__ == "__main__":

    tree = Tree()   
    tree.create_node("Harry",0,0, 0,0,0,"harry")  # root node
    tree.create_node("Jane",0,0, 0,0,0, "jane", parent = "harry")
    tree.create_node("Bill",0,0, 0,0,0, "bill", parent = "harry")
    tree.create_node("Grace",0,0, 0,0,0, "grace", parent = "bill")
    tree.create_node("Mark",0,0, 0,0,0, "mark", parent = "bill")
    tree.create_node("Joe",0,0, 0,0,0, "joe", parent = "jane")
    tree.create_node("Diane",0,0, 0,0,0, "diane", parent = "jane")
    tree.create_node("George",0,0, 0,0,0, "george", parent = "diane")
    tree.create_node("Mary",0,0, 0,0,0, "mary", parent = "diane")
    tree.create_node("Jill",0,0, 0,0,0, "jill", parent = "joe")
    tree.create_node("Carol",0,0, 0,0,0, "carol", parent = "joe")
    tree.create_node("Cat",0,0, 0,0,0, "cat", parent = "grace")
    tree.create_node("Mouse",0,0, 0,0,0, "mouse", parent = "grace")
    tree.create_node("Rat",0,0, 0,0,0, "rat", parent = "mark")
    tree.create_node("Bat",0,0, 0,0,0, "bat", parent = "mark")

    node = tree.__getitem__( "mouse")
    node.utility = 1.0
    print("="*80)
     
