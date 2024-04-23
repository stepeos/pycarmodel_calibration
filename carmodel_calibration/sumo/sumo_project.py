"""
file with classes that create the sumo project files like network, routes etc.
"""
import logging
import os
import re
from subprocess import Popen, PIPE, TimeoutExpired
from pathlib import Path
import xml.etree.ElementTree as ET
import warnings

from carmodel_calibration.fileaccess.parameter import ModelParameters
from carmodel_calibration.logging_config import configure_logging

configure_logging()
_LOGGER = logging.getLogger(__name__)

def _get_edge_number(edge):
    return int(float(re.findall(r"([\+\-]*\d*\.*\d+)",
                               str(edge.attrib["id"]))[0]))
class SumoProject:
    """class used to build sumo project for calibration purposes"""

    @staticmethod
    def create_sumo(project_path: Path, model: str, number_network_lanes: int):
        """
        create entire sumo project folder
        :param number_network_lanes:        count of lanes in the calibration
                                            grid, must be > 3
        """
        default_params = ModelParameters.get_defaults_dict()
        cfmodel = ModelParameters.create_parameter_set(".tmp/params", model,
                                                        **default_params)
        if not project_path.exists():
            project_path.mkdir(parents=True)
        network_path = project_path / "calibration_network.xml"
        routes_path = project_path / "calibration_routes.rou.xml"
        config_path = project_path / "calibration_config.sumocfg"
        # if number_network_lanes <= 2:
        #     raise ValueError("`number_network_lanes` must be atleast 3")
        SumoProject.create_network(2, max((3, number_network_lanes)),
                                   network_path)
        SumoProject.create_routes_from_network(network_path, routes_path,
                                               number_network_lanes)
        SumoProject.create_config(network_path, routes_path, config_path)
        SumoProject.write_followers_leader(routes_path,
                                           [cfmodel] * number_network_lanes)

    @staticmethod
    def create_network(x_number, y_number, file_path):
        """create grid network"""
        cmd = [
            "netgenerate", "--grid",
            "--default.speed=13.8889",
            f"--grid.x-number={x_number:d}",
            f"--grid.y-number={y_number:d}",
            "--grid.y-length=20",
            "--grid.x-length=2000",
            f"--output-file={str(file_path)}"
        ]
        _LOGGER.debug("executing:%s", " ".join(cmd))
        proc = Popen(cmd, stdout=PIPE)
        try:
            outs, errs = proc.communicate(timeout=10)
            _LOGGER.debug("%s \noutput:%s",
                         " ".join(cmd),
                         str(outs))
        except TimeoutExpired as exc:
            proc.kill()
            outs, errs = proc.communicate()
            _LOGGER.error("Network creation failed with %s and error %s",
                          outs,
                          errs)
            raise TimeoutError from exc

    @staticmethod
    def create_routes_from_network(network_path: Path, out_path: Path,
                                   number_used: int = None):
        """creates a routes file from the provided network"""
        pattern = r"^B\d*A\d*$"
        net_tree = ET.parse(str(network_path))
        net_root = net_tree.getroot()
        routes_root = ET.Element("routes")
        routes_root.set("xmlns:xsi",
                        "http://www.w3.org/2001/XMLSchema-instance")
        routes_root.set("xsi:noNamespaceSchemaLocation",
                        "http://sumo.dlr.de/xsd/routes_file.xsd")
        if out_path.exists():
            out_path.unlink()
        num_edges = 0
        for edge in net_root.findall('edge'):
            if re.match(pattern, str(edge.attrib['id'])):
                num_edges += 1
        edges =  net_root.findall('edge')
        edges = sorted(edges, key= lambda x: _get_edge_number(edge))
        for edge in edges:
            if re.match(pattern, str(edge.attrib['id'])):
                source = str(edge.attrib['id'])
                dest = source.replace('B', 'Z')
                dest = dest.replace('A', 'B')
                dest = dest.replace('Z', 'A')
                edge_number = _get_edge_number(edge)
                identification = f"t_{edge_number}"
                car_type = f"follower{edge_number:d}"
                if number_used:
                    if edge_number < number_used:
                        trip = ET.SubElement(routes_root, 'trip')
                        trip.set("id", identification)
                        trip.set("type", car_type)
                        trip.set("depart", "0.04")
                        trip.set("insertionChecks", "none")
                        trip.set("from", source)
                        trip.set("to", dest)
                        number = re.findall(r"(\d+)", source)[0]
                        if int(float(number)) == 0:
                            # <route id="r_BOTTOM" edges="B0A0 A0A1 A1A0 A0B0"/>
                            edges = [source, "A0A1", "A1A0", dest]
                        elif int(float(number)) == num_edges:
                            # <route id="r_TOP" edges="B249A249 A249A248 A248A249\
                            # A249B249"/>
                            edges = [f"{source}", f"A{num_edges-1}B{num_edges-2}",
                                    dest]
                        else:
                            # <route id="r_MIDDLE" edges="B145A145 A145B145"/>
                            edges = [source, dest]
                        route = ET.SubElement(routes_root, 'route')
                        identification = f"route_{edge_number}"
                        route.set("id", identification)
                        route.set("edges", " ".join(edges))
        routes_root[:] = sorted(routes_root, key=lambda child: child.tag)
        if hasattr(ET, "indent"):
            ET.indent(routes_root, "  ")
        routes_tree = ET.ElementTree(routes_root)
        routes_tree.write(str(out_path),
                          encoding="utf-8",
                          method="xml")

    @staticmethod
    def create_config(network_file: Path, routes_file: Path, out_path: Path):
        """creates as sumo config file from network and routes file"""
        if not (network_file.exists() and routes_file.exists()):
            if not network_file.exists():
                file_type = "network"
                file = str(network_file)
            else:
                file_type = "routes"
                file = str(routes_file)
            _LOGGER.error("%s-file not found at %s"
                            "\ncreate network-file first",
                            file_type,
                            str(file))
            raise FileNotFoundError
        cmd = [
            "sumo",
            "--step-length=0.04",
            "--collision.mingap-factor",
            "0.1",
            "-e",
            "5",
            "--net-file",
            f"{str(network_file)}",
            "--route-files",
            f"{str(routes_file)}",
            "-C",
            f"{str(out_path)}"
        ]
        my_env = os.environ.copy()
        my_env["SUMO_HOME"] = "/usr/share/sumo"
        _LOGGER.debug("executing:%s", " ".join(cmd))
        proc = Popen(cmd, stdout=PIPE, env=my_env)
        try:
            outs, errs = proc.communicate(timeout=10)
            _LOGGER.debug("%s \noutput:%s",
                         " ".join(cmd),
                         str(outs))
        except TimeoutExpired as exc:
            proc.kill()
            outs, errs = proc.communicate()
            _LOGGER.error("Sumo config creation failed with %s and error %s",
                          outs,
                          errs)
            raise TimeoutError from exc

    @staticmethod
    def write_followers_leader(routes_path, followers: list):
        """
        create 'follower0' from parameters and save to routes file
        if 'follower0' already exists, then it will be replaced
        :param followers:       list of `Parameters`
        """
        if not routes_path.exists():
            msg = ("Could not find routes-file. Create routes file first"
                   + "(file-name specified:%s)",
                   str(routes_path))
            _LOGGER.error(msg)
            raise FileNotFoundError(msg)
        routes_tree = ET.parse(str(routes_path))
        routes_root = routes_tree.getroot()
        for item in routes_root.findall('vType'):
            # if item.get('id') != 'leader0':
            routes_root.remove(item)
        max_id = 0
        for idx, follower in enumerate(followers):
            for item in routes_root.findall('trip'):
                if item.get("id") == f"t_{idx}":
                    speed_factor = follower.get_value("speedFactor")
                    if not speed_factor:
                        speed_factor = 1
                    item.set("speedFactor", str(speed_factor))
                    break
            follower_item = ET.Element("vType")
            follower_item.set("id", f"follower{idx}")
            found_desAccel = 0
            for key, value in follower.items():
                if key == "speedFactor":
                    continue
                elif (key == "desAccel1" or
                      key == "desAccel2" or
                      key == "desAccel3" or
                      key == "desAccel4" or
                      key == "desAccel5" or
                      key == "desAccel6"):
                    found_desAccel = 1
                    continue
                follower_item.set(key, str(value))
            if found_desAccel == 1:
                desAccel1 = follower.get_value("desAccel1")
                speed1 = 4 #5
                desAccel2 = follower.get_value("desAccel2")
                speed2 = 9 #12
                desAccel3 = follower.get_value("desAccel3")
                speed3 = 14 #20
                desAccel4 = follower.get_value("desAccel4")
                speed4 = 22 #30
                desAccel5 = follower.get_value("desAccel5")
                speed5 = 32 #40
                desAccel6 = follower.get_value("desAccel6")
                speed6 = 45 #50
                follower_item.set("desAccelProfile", str(speed1) + " "
                                   + str(desAccel1) + ","
                                   + str(speed2) + " " + str(desAccel2) + ","
                                   + str(speed3) + " " + str(desAccel3) + ","
                                   + str(speed4) + " " + str(desAccel4) + ","
                                   + str(speed5) + " " + str(desAccel5) + ","
                                   + str(speed6) + " " + str(desAccel6))
            routes_root.insert(idx, follower_item)
            max_id = idx
        leader = ET.Element("vType")
        leader.set("id", "leader0")
        routes_root.insert(max_id + 1, leader)
        if hasattr(ET, "indent"):
            ET.indent(routes_root, "  ")
        routes_tree.write(str(routes_path))

    @staticmethod
    def get_number_routes(routes_path):
        """returns the number of routes in a routes file"""
        routes_tree = ET.parse(str(routes_path))
        routes_root = routes_tree.getroot()
        return len(routes_root.findall("route"))

    @staticmethod
    def get_number_followers(routes_path):
        """returns the number of routes in a routes file"""
        routes_tree = ET.parse(str(routes_path))
        routes_root = routes_tree.getroot()
        followers = 0
        for element in routes_root.findall("vType"):
            if element.attrib['id'].startswith("follower"):
                followers += 1
        return followers

    @staticmethod
    def get_network_info(network_file):
        """returns the necessary network info"""
        network_info = {}
        network_tree = ET.parse(network_file)
        network_root = network_tree.getroot()
        pattern = r"^B\d*A\d*$"
        for edge in network_root.findall("edge"):
            if re.match(pattern, str(edge.attrib['id'])):
                edge_numbers = (re.findall(r"([\+\-]*\d*\.*\d+)",
                               str(edge.attrib["id"])))
                edge_numbers = [int(float(item)) for item in edge_numbers]
                if edge_numbers[0] != edge_numbers[1]:
                    print(str(edge.attrib['id']))
                    continue
                edge_number = edge_numbers[0]
                network_info[edge_number] = str(edge.attrib['id'])
        return network_info