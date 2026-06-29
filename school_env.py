from ipaddress import IPv4Network, IPv4Address # Fix lỗi NameError
from CybORG.Shared import Scenario
from CybORG.Shared.Scenarios.ScenarioGenerator import ScenarioGenerator
from CybORG.Simulator.Subnet import Subnet
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Interface import Interface
from CybORG.Agents import BaseAgent
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
from CybORG.Shared.Session import RedAbstractSession # Import session chuẩn cho Red

class CustomBlueAgent(BaseAgent):
    def __init__(self, name=None): super().__init__(name)
    def train(self, results): pass
    def get_action(self, observation, action_space): pass
    def end_episode(self): pass
    def set_initial_state(self, observation, action_space): pass
    def set_initial_values(self, action_space, observation): pass

class AgentConfig:
    def __init__(self, name, team, agent_obj):
        self.name = name
        self.team = team
        self.agent_type = agent_obj
        self.starting_sessions = []
        self.default_actions = None 
        self.actions = []
        self.active = True
        self.internal_only = False
        self.allowed_subnets = ["Admin", "Student", "Server"]
        self.osint = {}

class SchoolScenarioGenerator(ScenarioGenerator):
    def __init__(self):
        super().__init__()
        self.update_each_step = True
        self.step_limit = 200
        self.MESSAGE_LENGTH = 100 # Cần cho SimulationController

    def make_host(self, name, subnet_obj, ip, np_random):
        system_info = {"OSType": "Linux", "OSDistribution": "Ubuntu", "OSVersion": "20.04", "Architecture": "x64"}
        interface = Interface(name=f"{name}_eth0", ip_address=ip, subnet=str(subnet_obj.cidr))
        return Host(np_random=np_random, system_info=system_info, hostname=name, interfaces=[interface])

    def create_scenario(self, np_random):
        subnets = {
            "Admin": Subnet(name="Admin", cidr=IPv4Network("10.0.0.0/24")),
            "Student": Subnet(name="Student", cidr=IPv4Network("10.0.1.0/24")),
            "Server": Subnet(name="Server", cidr=IPv4Network("10.0.2.0/24")),
        }

        hosts_dict = {
            "Admin_PC": self.make_host("Admin_PC", subnets["Admin"], "10.0.0.10", np_random),
            "Teacher_PC": self.make_host("Teacher_PC", subnets["Admin"], "10.0.0.11", np_random),
            "Student_PC1": self.make_host("Student_PC1", subnets["Student"], "10.0.1.10", np_random),
            "Student_PC2": self.make_host("Student_PC2", subnets["Student"], "10.0.1.11", np_random),
            "File_Server": self.make_host("File_Server", subnets["Server"], "10.0.2.10", np_random),
            "Web_Server": self.make_host("Web_Server", subnets["Server"], "10.0.2.20", np_random),
        }

        agents = {
            "Blue": AgentConfig("Blue", "Blue", CustomBlueAgent(name="Blue")),
            "Red": AgentConfig("Red", "Red", FiniteStateRedAgent(name="Red"))
        }

        # Cấu hình Red Session
        red_session = RedAbstractSession(
            ident=0, 
            hostname="Student_PC1", 
            username="user", 
            agent="Red", 
            pid=1234, 
            session_type="red_abstract_session"
        )
        
        # 🔥 QUAN TRỌNG: Nạp sẵn các máy mục tiêu vào bộ nhớ của Red
        # Điều này giúp Red Agent thấy các máy khác để tấn công ngay từ bước Check Env
        target_ips = [
            IPv4Address("10.0.1.11"), # Student_PC2
            IPv4Address("10.0.2.10"), # File_Server
            IPv4Address("10.0.2.20")  # Web_Server
        ]
        for ip in target_ips:
            red_session.addport(ip, 80) # Giả định Red đã tìm thấy các IP này

        agents["Red"].starting_sessions = [red_session]
        
        # OSINT dùng Hostname làm key để SimulationController nhận diện
        agents["Red"].osint = {
            'Hosts': {
                "Student_PC1": {"System info": "All", "Interfaces": "All"},
                "Student_PC2": {"Interfaces": "All"},
                "File_Server": {"Interfaces": "All"}
            }
        }

        return Scenario(
            hosts=hosts_dict, 
            agents=agents, 
            subnets=subnets,
            team_agents={"Blue": ["Blue"], "Red": ["Red"]}
        )

    def determine_done(self, env_controller):
        return env_controller.step_count >= self.step_limit

    def validate_scenario(self, scenario): return True