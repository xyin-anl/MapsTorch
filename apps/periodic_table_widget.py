import anywidget
import traitlets


class PeriodicTableWidget(anywidget.AnyWidget):
    # Synced properties between Python and JavaScript
    selected_elements = traitlets.Dict({}).tag(sync=True)
    disabled_elements = traitlets.List([]).tag(sync=True)
    states = traitlets.Int(1).tag(sync=True)
    disabled_color = traitlets.Unicode("rgb(255, 255, 255)").tag(sync=True)  # white
    unselected_color = traitlets.Unicode("rgb(215, 215, 215)").tag(
        sync=True
    )  # light silver
    selected_colors = traitlets.List(["rgb(191, 219, 254)"]).tag(sync=True)  # blue-200
    on_change = traitlets.Dict({}).tag(sync=True)  # To handle selection changes

    # Widget front-end JavaScript code
    _esm = """
    const elementTable = [
      ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
      ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
      ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
      ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
      ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
      ['Cs', 'Ba', '*', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
      ['Fr', 'Ra', '#', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
      ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
      ['', '', '*', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
      ['', '', '#', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    ];

    function createElementDiv(element, model) {
      const div = document.createElement('div');

      if (element === '' || element === '*' || element === '#') {
        div.style.width = '38px';
        div.style.height = '38px';
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.justifyContent = 'center';
        div.style.fontSize = '0.875rem';
        div.style.userSelect = 'none';
        div.textContent = element;
        return div;
      }

      div.style.width = '38px';
      div.style.height = '38px';
      div.style.display = 'flex';
      div.style.alignItems = 'center';
      div.style.justifyContent = 'center';
      div.style.fontSize = '0.875rem';
      div.style.borderRadius = '0.25rem';
      div.style.transition = 'all 0.2s';
      div.style.userSelect = 'none';
      div.style.border = 'none';
      div.textContent = element;

      const updateElementStyle = () => {
        const selected = model.get('selected_elements');
        const disabled = new Set(model.get('disabled_elements'));

        if (disabled.has(element)) {
          div.style.backgroundColor = model.get('disabled_color');
          div.style.cursor = 'not-allowed';
        } else {
          const colors = model.get('selected_colors');
          div.style.backgroundColor = element in selected ? 
            colors[selected[element]] : 
            model.get('unselected_color');
          div.style.cursor = 'pointer';
        }
      };

      if (!model.get('disabled_elements').includes(element)) {
        div.addEventListener('mouseover', () => {
          div.style.filter = 'brightness(0.9)';
        });

        div.addEventListener('mouseout', () => {
          div.style.filter = 'brightness(1)';
        });

        div.addEventListener('click', () => {
          const currentSelected = model.get('selected_elements');
          const newSelected = {...currentSelected};

          if (element in newSelected) {
            if (newSelected[element] < model.get('states') - 1) {
              newSelected[element]++;
            } else {
              delete newSelected[element];
            }
          } else {
            newSelected[element] = 0;
          }

          model.set('selected_elements', newSelected);
          model.save_changes();

          model.set('on_change', { 
            selected: Object.keys(newSelected),
            states: Object.entries(newSelected).map(([el, state]) => ({element: el, state}))
          });
          model.save_changes();
        });
      }

      // Initial style
      requestAnimationFrame(() => {
        updateElementStyle();
      });

      // Listen for changes
      model.on('change:selected_elements', updateElementStyle);
      model.on('change:disabled_elements', updateElementStyle);
      model.on('change:selected_colors', updateElementStyle);

      return div;
    }

    function render({ model, el }) {
      // Create container
      const container = document.createElement('div');
      container.style.display = 'inline-block';
      container.style.padding = '1rem';
      container.style.backgroundColor = 'white';
      container.style.borderRadius = '0.5rem';
      container.style.boxShadow = '0 1px 3px 0 rgba(0, 0, 0, 0.1)';

      // Create grid container
      const grid = document.createElement('div');
      grid.style.display = 'grid';
      grid.style.gap = '4px';
      grid.style.gridTemplateColumns = 'repeat(18, 38px)';

      // Create table
      elementTable.forEach(row => {
        row.forEach(element => {
          const elementDiv = createElementDiv(element, model);
          grid.appendChild(elementDiv);
        });
      });

      container.appendChild(grid);

      // Create selection display
      const selectionDiv = document.createElement('div');
      selectionDiv.style.marginTop = '1rem';
      selectionDiv.style.fontSize = '0.875rem';
      selectionDiv.style.color = '#4B5563';

      function updateSelection() {
        const selected = model.get('selected_elements');
        selectionDiv.textContent = 'Selected: ' + 
          Object.entries(selected)
            .map(([el, state]) => `${el}`)
            .join(', ');
      }

      updateSelection();
      model.on('change:selected_elements', updateSelection);

      container.appendChild(selectionDiv);
      el.appendChild(container);
    }

    export default { render };
    """

    _css = ""

    def __init__(
        self,
        initial_selected=None,
        initial_disabled=None,
        states=1,
        disabled_color=None,
        unselected_color=None,
        selected_colors=None,
        on_selection_change=None,
    ):
        """Initialize the periodic table widget."""
        # Initialize traitlets first
        super().__init__()

        # Set number of states
        self.states = states

        # Set disabled elements first
        self.disabled_elements = list(initial_disabled or [])

        # Set colors
        if disabled_color is not None:
            self.disabled_color = disabled_color
        if unselected_color is not None:
            self.unselected_color = unselected_color

        # Set default colors for states if not provided
        if selected_colors is not None:
            self.selected_colors = selected_colors[:states]
        else:
            default_colors = [
                "rgb(191, 219, 254)",  # blue-200
                "rgb(147, 197, 253)",  # blue-300
                "rgb(96, 165, 250)",  # blue-400
            ]
            self.selected_colors = default_colors[:states]

        # Set initial selection state
        if initial_selected:
            self.selected_elements = {
                element: state
                for element, state in initial_selected.items()
                if state < states and element not in self.disabled_elements
            }

        # Set up callback if provided
        if on_selection_change:
            self.observe(
                lambda change: on_selection_change(change["new"]), names=["on_change"]
            )
