import taichi as ti

@ti.data_oriented
class FLIP_2D:

    def __init__(self, dt, grid_x, grid_y, width, height, tot_point):

        self.density = 1000.0
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.dx = ti.max(width/grid_x, height/grid_y)
        self.inv_dx = 1.0 / self.dx
        self.tot_grid = grid_x * grid_y
        self.tot_point = tot_point
        self.dt = dt

        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=tot_point)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=tot_point)
        self.rest_density = ti.field(dtype=ti.f32, shape=())
        self.radius = 0.3 * self.dx
        self.p_inv = 1.0 / (2.2 * self.radius)
        self.pn_x = int(width * self.p_inv + 1)
        self.pn_y = int(height * self.p_inv + 1)
        self.pn_cell = self.pn_x * self.pn_y
        self.cnt_p = ti.field(dtype=ti.i32, shape=self.pn_cell)
        self.pcell_first = ti.field(dtype=ti.i32, shape=self.pn_cell + 1)
        self.pcell2p = ti.field(dtype=ti.i32, shape=tot_point)

        self.grid_v = ti.Vector.field(2, dtype=ti.f32, shape=self.tot_grid)
        self.grid_v_old = ti.Vector.field(2, dtype=ti.f32, shape=self.tot_grid)
        self.prev_grid_v = ti.Vector.field(2, dtype=ti.f32, shape=self.tot_grid)
        self.grid_dv = ti.Vector.field(2, dtype=ti.f32, shape=self.tot_grid)
        self.status = ti.field(dtype=ti.f32, shape=self.tot_grid)
        self.cell_type = ti.field(dtype=ti.i32, shape=self.tot_grid)
        self.cell_density = ti.field(dtype=ti.f32, shape=self.tot_grid)
        self.pressures = ti.field(dtype=ti.f32, shape=(grid_x, grid_y))
        self.new_pressures = ti.field(dtype=ti.f32, shape=(grid_x, grid_y))
        self.div = ti.field(dtype=ti.f32, shape=(grid_x, grid_y))

    @ti.kernel
    def step_particle(self):
        for i in self.pos:
            self.vel[i].y -= self.dt * 9.8
            self.pos[i] += self.vel[i] * self.dt

    @ti.func
    def clamp(self, x, l, r):
        return ti.max(ti.min(x, r), l)

    @ti.kernel
    def prepare_kernel1(self):
        self.cnt_p.fill(0)
        for i in self.pos:
            xi = self.clamp(ti.cast(self.pos[i].x * self.p_inv, ti.i32), 0, self.pn_x - 1)
            yi = self.clamp(ti.cast(self.pos[i].y * self.p_inv, ti.i32), 0, self.pn_y - 1)
            idx = xi * self.pn_y + yi
            self.cnt_p[idx] += 1


    @ti.kernel
    def prepare_kernel2(self):
        for i in self.pos:
            xi = self.clamp(ti.cast(self.pos[i].x * self.p_inv, ti.i32), 0, self.pn_x - 1)
            yi = self.clamp(ti.cast(self.pos[i].y * self.p_inv, ti.i32), 0, self.pn_y - 1)
            idx = xi * self.pn_y + yi
            m = ti.atomic_add(self.pcell_first[idx], -1)
            self.pcell2p[m] = i

    def pushParticlesApart_prepare(self):
        self.prepare_kernel1()
        first = 0
        for i in range(self.pn_cell):
            first += self.cnt_p[i]
            self.pcell_first[i] = first
        self.pcell_first[self.pn_cell] = first
        self.prepare_kernel2()

    @ti.kernel
    def pushParticlesApart_kernel(self):
        min_d = 2.0 * self.radius
        min_d2 = min_d * min_d

        for i in self.pos:
            px = self.pos[i].x
            py = self.pos[i].y
            pxi = ti.cast(px * self.p_inv, ti.i32)
            pyi = ti.cast(py * self.p_inv, ti.i32)
            x0 = ti.max(pxi - 1, 0)
            x1 = ti.min(pxi + 1, self.pn_x - 1)
            y0 = ti.max(pyi - 1, 0)
            y1 = ti.min(pyi + 1, self.pn_y - 1)

            for xi in range(x0, x1+1):
                for yi in range(y0, y1+1):
                    idx = xi * self.pn_x + yi
                    l = self.pcell_first[idx]
                    r = self.pcell_first[idx + 1]
                    for j in range(l, r):
                        pid = self.pcell2p[j]
                        if i == j:
                            continue
                        qx = self.pos[pid].x
                        qy = self.pos[pid].y

                        dx = qx - px
                        dy = qy - py
                        d2 = dx * dx + dy * dy
                        if (d2 > min_d2):
                            continue
                        d = ti.math.sqrt(d2)
                        s = 0.5 * (min_d - d) / d;
                        dx *= s
                        dy *= s
                        self.pos[i].x -= dx
                        self.pos[i].y -= dy
                        self.pos[pid].x += dx
                        self.pos[pid].y += dy
                        # when I get here, I suddenly realize that, this process cannot be paralleled, so I stop it

    # this can't be parallel!!!!!!!!!
    def pushParticlesApart(self, iters):
        self.pushParticlesApart_prepare()
        for i in range(iters):
            self.pushParticlesApart_kernel()

    @ti.kernel
    def handleParticleCollisions(self, obstacleX: ti.f32, obstacleY: ti.f32, obstacleRadius: ti.f32):

        minDist = obstacleRadius + self.radius
        minDist2 = minDist * minDist

        minX = self.dx + self.radius
        maxX = (self.grid_x - 1) * self.dx - self.radius
        minY = self.dx + self.radius
        maxY = (self.grid_y - 1) * self.dx - self.radius

        for i in self.pos:
            x = self.pos[i].x
            y = self.pos[i].y
            dx = x - obstacleX
            dy = y - obstacleY
            d2 = dx * dx + dy * dy

            if d2 < minDist2:
                d = ti.sqrt(d2)
                s = (minDist - d) / d
                x += dx * s
                y += dy * s

            if (x < minX):
                x = minX
                self.vel[i].x = 0
            if (x > maxX):
                x = maxX
                self.vel[i].x = 0
            if (y < minY):
                y = minY
                self.vel[i].y = 0
            if (y > maxY):
                y = maxY
                self.vel[i].y = 0

            self.pos[i].x = x
            self.pos[i].y = y


    @ti.kernel
    def updateParticleDensity(self):

        n = self.grid_y
        h = self.dx
        h1 = self.inv_dx
        h2 = 0.5 * h

        self.cell_density.fill(0.0);

        for i in self.pos:
            x = self.clamp(self.pos[i].x, self.dx, (self.grid_x - 1) * self.dx)
            y = self.clamp(self.pos[i].y, self.dx, (self.grid_y - 1) * self.dx)

            x0 = ti.cast((x - h2) * h1, ti.i32)
            tx = ((x - h2) - x0 * h) * h1
            x1 = ti.min(x0 + 1, self.grid_x - 2)

            y0 = ti.cast((y - h2) * h1, ti.i32)
            ty = ((y - h2) - y0 * h) * h1
            y1 = ti.min(y0 + 1,self.grid_y - 2)

            sx = 1.0 - tx
            sy = 1.0 - ty

            if (x0 < self.grid_x and y0 < self.grid_y):
                self.cell_density[x0 * n + y0] += sx * sy
            if (x1 < self.grid_x and y0 < self.grid_y):
                self.cell_density[x1 * n + y0] += tx * sy
            if (x1 < self.grid_x and y1 < self.grid_y):
                self.cell_density[x1 * n + y1] += tx * ty
            if (x0 < self.grid_x and y1 < self.grid_y):
                self.cell_density[x0 * n + y1] += sx * ty

        if (self.rest_density[None] == 0.0):
            sum = 0.0
            numFluidCells = 0.0
            for i in range(self.tot_grid):
                if (self.cell_type[i] == 1):
                    sum += self.cell_density[i]
                    numFluidCells += 1.0
            if (numFluidCells > 0):
                self.rest_density[None] = sum / numFluidCells

    @ti.func
    def isvalid(self, idx, offset):
        valid = 1.0
        if self.cell_type[idx] == 2 and self.cell_type[idx - offset] == 2:
            valid = 0.0
        return valid

    @ti.kernel
    def transfer_V(self, flipRatio: ti.f32):
        n = self.grid_y
        h = self.dx
        h1 = self.inv_dx
        h2 = 0.5 * h

        delta_x = ti.Vector([0.0, h2])
        delta_y = ti.Vector([h2, 0.0])

        for i in self.pos:
            for j in ti.static(range(2)):

                dx = delta_x[j]
                dy = delta_y[j]
                x = self.clamp(self.pos[i].x, self.dx, (self.grid_x - 1) * self.dx)
                y = self.clamp(self.pos[i].y, self.dx, (self.grid_y - 1) * self.dx)

                x0 = ti.min(ti.cast((x - dx) * h1, ti.i32), self.grid_x - 2)
                tx = ((x - h2) - x0 * h) * h1
                x1 = ti.min(x0 + 1, self.grid_x - 2)

                y0 = ti.min(ti.cast((y - dy) * h1, ti.i32), self.grid_y - 2)
                ty = ((y - h2) - y0 * h) * h1
                y1 = ti.min(y0 + 1, self.grid_y - 2)

                sx = 1.0 - tx
                sy = 1.0 - ty

                d0 = sx*sy
                d1 = tx*sy
                d2 = tx*ty
                d3 = sx*ty

                nr0 = x0*n + y0
                nr1 = x1*n + y0
                nr2 = x1*n + y1
                nr3 = x0*n + y1

                offset = n - j * (n - 1)
                valid0 = self.isvalid(nr0, offset)
                valid1 = self.isvalid(nr1, offset)
                valid2 = self.isvalid(nr2, offset)
                valid3 = self.isvalid(nr3, offset)

                v = self.vel[i][j]
                d = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3

                if (d > 0.0):
                    picV = (valid0 * d0 * self.grid_v[nr0][j] + valid1 * d1 * self.grid_v[nr1][j] + valid2 * d2 * self.grid_v[nr2][j] + valid3 * d3 * self.grid_v[nr3][j]) / d
                    corr = (valid0 * d0 * (self.grid_v[nr0][j] - self.prev_grid_v[nr0][j]) + valid1 * d1 * (self.grid_v[nr1][j] - self.prev_grid_v[nr1][j])
                        + valid2 * d2 * (self.grid_v[nr2][j] - self.prev_grid_v[nr2][j]) + valid3 * d3 * (self.grid_v[nr3][j] - self.prev_grid_v[nr3][j])) / d
                    flipV = v + corr
                    self.vel[i][j] = (1.0 - flipRatio) * picV + flipRatio * flipV

    @ti.kernel
    def transfer_V_toGrid(self):
        n = self.grid_y
        h = self.dx
        h1 = self.inv_dx
        h2 = 0.5 * h
        self.grid_v.fill(0)
        self.grid_dv.fill(0)

        for i in range(self.tot_grid):
            if self.status[i] == 0.0:
                self.cell_type[i] = 3
            else:
                self.cell_type[i] = 2

        for i in self.pos:
            xi = self.clamp(ti.cast(self.pos[i].x * h1, ti.i32), 0, self.grid_x - 1)
            yi = self.clamp(ti.cast(self.pos[i].y * h1, ti.i32), 0, self.grid_y - 1)
            idx = xi * n + yi
            if self.cell_type[idx] == 2:
                self.cell_type[idx] = 1

        delta_x = ti.Vector([0.0, h2])
        delta_y = ti.Vector([h2, 0.0])

        for i in self.pos:
            for j in ti.static(range(2)):
                dx = delta_x[j]
                dy = delta_y[j]
                x = self.clamp(self.pos[i].x, self.dx, (self.grid_x - 1) * self.dx)
                y = self.clamp(self.pos[i].y, self.dx, (self.grid_y - 1) * self.dx)

                x0 = ti.min(ti.cast((x - dx) * h1, ti.i32), self.grid_x - 2)
                tx = ((x - h2) - x0 * h) * h1
                x1 = ti.min(x0 + 1, self.grid_x - 2)

                y0 = ti.min(ti.cast((y - dy) * h1, ti.i32), self.grid_y - 2)
                ty = ((y - h2) - y0 * h) * h1
                y1 = ti.min(y0 + 1, self.grid_y - 2)

                sx = 1.0 - tx
                sy = 1.0 - ty

                d0 = sx * sy
                d1 = tx * sy
                d2 = tx * ty
                d3 = sx * ty

                nr0 = x0 * n + y0
                nr1 = x1 * n + y0
                nr2 = x1 * n + y1
                nr3 = x0 * n + y1

                pv = self.vel[i][j]
                self.grid_v[nr0][j] += pv * d0
                self.grid_dv[nr0][j] += d0
                self.grid_v[nr1][j] += pv * d1
                self.grid_dv[nr1][j] += d1
                self.grid_v[nr2][j] += pv * d2
                self.grid_dv[nr2][j] += d2
                self.grid_v[nr3][j] += pv * d3
                self.grid_dv[nr3][j] += d3

        for i in self.grid_v:
            if self.grid_dv[i][0] > 0:
                self.grid_v[i][0] /= self.grid_dv[i][0]
            if self.grid_dv[i][1] > 0:
                self.grid_v[i][1] /= self.grid_dv[i][1]

        for i, j in ti.ndrange(self.grid_x, self.grid_y):
            xx = i * n + j
            if self.cell_type[xx] == 3:
                self.grid_v[xx] = self.prev_grid_v[xx]
            if i > 0 and self.cell_type[(i - 1) * n + j] == 3:
                self.grid_v[xx].x = 0.0
            if j > 0 and self.cell_type[i * n + j - 1] == 3:
                self.grid_v[xx].y = 0.0

    # still can't parallel !!!!!
    def kernel_3(self, overRelaxation: ti.f32):
        n = self.grid_y
        for ii, jj in ti.ndrange(self.grid_x - 2, self.grid_y - 2):
            i = ii + 1
            j = jj + 1
            if self.cell_type[i * n + j] != 1:
                continue

            center = i * n + j
            left = (i - 1) * n + j
            right = (i + 1) * n + j
            bottom = i * n + j - 1
            top = i * n + j + 1

            sx0 = self.status[left]
            sx1 = self.status[right]
            sy0 = self.status[bottom]
            sy1 = self.status[top]
            s = sx0 + sx1 + sy0 + sy1
            if (s == 0.0):
                continue

            div = self.grid_v[right].x - self.grid_v[center].x + self.grid_v[top].y - self.grid_v[center].y

            # if (self.rest_density[None] > 0.0):
            #     compression = self.cell_density[i * n + j] - self.rest_density[None]
            #     if (compression > 0.0):
            #         div = div - compression

            p = -div / s
            p *= overRelaxation

            self.grid_v[center].x -= sx0 * p
            self.grid_v[right].x += sx1 * p
            self.grid_v[center].y -= sy0 * p
            self.grid_v[top].y += sy1 * p

    def solveIncompressibility(self, numIters, overRelaxation):

        self.prev_grid_v.copy_from(self.grid_v)

        n = self.grid_y
        cp = self.density * self.dx / self.dt

        for iter in range(numIters):
            # self.grid_v_old.copy_from(self.grid_v)
            self.kernel_3(overRelaxation)

    @ti.func
    def is_solid(self, i, j):
        return self.cell_type[i * self.grid_y + j] == 3

    @ti.func
    def is_air(self, i, j):
        return self.cell_type[i * self.grid_y + j] == 2

    @ti.func
    def is_fluid(self, i, j):
        return self.cell_type[i * self.grid_y + j] == 1

    @ti.kernel
    def fill_matrix(self, A: ti.types.sparse_matrix_builder(), F_b: ti.types.ndarray()):
        m_g = self.grid_y
        for I in ti.grouped(self.div):
            F_b[I[0] * m_g + I[1]] = - self.div[I] * self.dx * self.dx * self.density / self.dt

        for i, j in ti.ndrange(m_g, m_g):
            I = i * m_g + j
            if self.is_fluid(i, j):
                if not self.is_solid(i - 1, j):
                    A[I, I] += 1.0
                    if not self.is_air(i - 1, j):
                        A[I - m_g, I] -= 1.0

                if not self.is_solid(i + 1, j):
                    A[I, I] += 1.0
                    if not self.is_air(i + 1, j):
                        A[I + m_g, I] -= 1.0

                if not self.is_solid(i, j - 1):
                    A[I, I] += 1.0
                    if not self.is_air(i, j - 1):
                        A[I, I - 1] -= 1.0

                if not self.is_solid(i, j + 1):
                    A[I, I] += 1.0
                    if not self.is_air(i, j + 1):
                        A[I, I + 1] -= 1.0
            else:
                # if is_solid(i, j) or is_air(i, j)
                A[I, I] += 1.0
                F_b[I] = 0.0

    @ti.kernel
    def copy_pressure(self, p_in: ti.types.ndarray(), p_out: ti.template()):
        for I in ti.grouped(p_out):
            p_out[I] = p_in[I[0] * self.grid_y + I[1]]

    @ti.kernel
    def projection(self):
        for i, j in ti.ndrange(self.grid_x, self.grid_y):
            if self.is_fluid(i, j):
                grad_p = ti.Vector([self.pressures[i + 1, j] - self.pressures[i, j], self.pressures[i, j + 1] - self.pressures[i, j]]) / self.dx
                if self.is_solid(i + 1, j):
                    grad_p.x = (self.pressures[i, j] - self.pressures[i - 1, j]) / self.dx
                if self.is_solid(i, j + 1):
                    grad_p.y = (self.pressures[i, j] - self.pressures[i, j - 1]) / self.dx
                self.grid_v[i * self.grid_y + j] -= grad_p / self.density * self.dt

        # for i, j in ti.ndrange(self.grid_x + 1, self.grid_x):
        #     if self.is_fluid(i - 1, j) or self.is_fluid(i, j):
        #         if self.is_solid(i - 1, j) or self.is_solid(i, j):
        #             self.grid_v[i * self.grid_y + j].x = 0.0
        #         else:
        #             self.grid_v[i * self.grid_y + j].x -= (self.pressures[i, j] - self.pressures[i - 1, j]) / self.dx / self.density * self.dt
        #
        # for i, j in ti.ndrange(self.grid_x, self.grid_x + 1):
        #     if self.is_fluid(i, j - 1) or self.is_fluid(i, j):
        #         if self.is_solid(i, j - 1) or self.is_solid(i, j):
        #             self.grid_v[i * self.grid_y + j].y = 0.0
        #         else:
        #             self.grid_v[i * self.grid_y + j].y -= (self.pressures[i, j] - self.pressures[i, j - 1]) / self.dx / self.density * self.dt

    @ti.kernel
    def solve_divergence(self):

        for i, j in self.div:
            if self.is_fluid(i, j):
                v_l = self.grid_v[i * self.grid_y + j].x
                v_r = self.grid_v[(i + 1) * self.grid_y + j].x
                v_d = self.grid_v[i * self.grid_y + j].y
                v_u = self.grid_v[i * self.grid_y + j + 1].y

                div = v_r - v_l + v_u - v_d

                if self.is_solid(i - 1, j):
                    div += v_l
                if self.is_solid(i + 1, j):
                    div -= v_r
                if self.is_solid(i, j - 1):
                    div += v_d
                if self.is_solid(i, j + 1):
                    div -= v_u

                self.div[i, j] = div / self.dx / 2.0

    @ti.kernel
    def pressure_jacobi(self, p: ti.template(), new_p: ti.template()):

        w = 0.95

        for i, j in p:
            if self.is_fluid(i, j):

                p_l = 0.0
                p_r = 0.0
                p_d = 0.0
                p_u = 0.0

                k = 4
                if self.is_solid(i - 1, j):
                    p_l = 0.0
                    k -= 1
                else:
                    p_l = p[i - 1, j]

                if self.is_solid(i + 1, j):
                    p_r = 0.0
                    k -= 1
                else:
                    p_r = p[i + 1, j]

                if self.is_solid(i, j - 1):
                    p_d = 0.0
                    k -= 1
                else:
                    p_d = p[i, j - 1]

                if self.is_solid(i, j + 1):
                    p_u = 0.0
                    k -= 1
                else:
                    p_u = p[i, j + 1]

                new_p[i, j] = (1 - w) * p[i, j] + w * (
                            p_l + p_r + p_d + p_u - self.div[i, j] * self.density / self.dt * (self.dx * self.dx)) / k

    def solveIncompressibility_parallel(self):
        self.prev_grid_v.copy_from(self.grid_v)
        self.solve_divergence()
        N = self.tot_grid
        K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
        F_b = ti.ndarray(ti.f32, shape=N)
        self.fill_matrix(K, F_b)
        L = K.build()
        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(L)
        solver.factorize(L)
        p = solver.solve(F_b)
        self.copy_pressure(p, self.pressures)

        # for i in range(20):
        #     self.pressure_jacobi(self.pressures, self.new_pressures)
        #     self.pressures.copy_from(self.new_pressures)

        self.projection()

    def simulate_step(self, flipRatio, numPressureIters, numParticleIters, overRelaxation, obstacleX, abstacleY, obstacleRadius):

        self.step_particle()
        # self.pushParticlesApart(numParticleIters)
        self.handleParticleCollisions(obstacleX, abstacleY, obstacleRadius)
        self.prev_grid_v.copy_from(self.grid_v)
        self.updateParticleDensity()
        self.transfer_V_toGrid()


        for i in range(30, 40):
            for j in range(70, 80):
                print(f"{self.grid_v[i * 100 + j].y:.4f}", end=" ")
            print("")
        self.solveIncompressibility_parallel()
        # self.solveIncompressibility(numParticleIters, overRelaxation)
        for i in range(30, 40):
            for j in range(70, 80):
                print(f"{self.grid_v[i * 100 + j].y:.4f}", end=" ")
            print("")
        self.transfer_V(flipRatio)
        # print(self.vel[20])


