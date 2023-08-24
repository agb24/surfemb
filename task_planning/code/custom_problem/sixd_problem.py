import time
import numpy as np
import pandas as pd

from permutandis.solvers.mcts import MCTSSolver
from permutandis.solvers.baseline import BaselineSolver

from verifier import RadiusCollisionChecker, WorkspaceChecker


class SampleIntermediate():
    def __init__(self, det_obj_ids=[0]):
        self.obj_ids = det_obj_ids
        sixd_df = pd.read_pickle("/home/ise.ros/akshay_work/NN_Implementations/surfemb/task_planning/6d_pose.pkl")
        self.filter_df = sixd_df[ sixd_df["true_obj_id"].isin(det_obj_ids) ]


class CustomProblem():
    def __init__(self, src, tgt, workspace):
        """
        Class for 6D Pose Rearrangement, 
        """
        self.src = src
        self.tgt = tgt
        self.workspace = workspace

    def assert_solution_valid(self, actions):
        """
        Ensure validity by: 
            1.Collision Check  
            2.Workspace Limit check 
        """
        cur = self.src.copy()
        tgt = self.tgt
        collisions = RadiusCollisionChecker(radius=self.radius)

        def collision_check(s):
            for n, x in enumerate(s):
                coll_ = collisions(s, x)
                coll_[n] = False
                if np.any(coll_):
                    return False
            return True

        workspace_check = WorkspaceChecker(workspace=self.workspace)
        assert collision_check(cur) and collision_check(tgt)
        assert workspace_check(cur) and workspace_check(tgt)
        for (object_id, place_pos) in actions:
            assert workspace_check(place_pos)
            cur[object_id] = place_pos
        d = np.linalg.norm(cur - tgt, axis=-1)
        cond = d <= 1e-3
        assert np.all(cond), f'Some objects are not at target.'
        return


def make_problem(problem):
    # 
    solver = MCTSSolver(max_iterations=10000, nu=1.0)
    outputs = solver(problem)



