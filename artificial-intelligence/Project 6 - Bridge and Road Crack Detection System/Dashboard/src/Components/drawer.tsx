import { Link } from "react-router-dom";
import * as React from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import Button from "@mui/material/Button";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import menuIcon from "../assets/menuIcon.svg";




export default function TheDrawer() {
  const [open, setOpen] = React.useState(false);

  const toggleDrawer = (newOpen: boolean) => () => {
    setOpen(newOpen);
  };

  const DrawerList = (
    <Box sx={{ width: 250, color: 'white' }} role="presentation" onClick={toggleDrawer(false)}>
      <List>
        <ListItem disablePadding>
          <ListItemButton
            component={Link}
            to="/"
            onClick={() => setOpen(false)}
          >
            <ListItemText primary="Home" />
          </ListItemButton>
        </ListItem>

        <ListItem disablePadding>
          <ListItemButton
            component={Link}
            to="/uploader"
            onClick={() => setOpen(false)}
          >
            <ListItemText primary="Upload Single File" />
          </ListItemButton>
        </ListItem>

        <ListItem disablePadding>
          <ListItemButton
            component={Link}
            to="/Uploadmultiplefiles"
            onClick={() => setOpen(false)}
          >
            <ListItemText primary="Upload Multiple Files" />
          </ListItemButton>
        </ListItem>

      </List>
    </Box>
  );

  return (
    <div style={{ position: "fixed", top: 16, left: 16, zIndex: 1000 }}>
      <Button onClick={toggleDrawer(true)}>
        <img src={menuIcon} alt="icon" style={{ width: 24, height: 24 }} />
      </Button>
      <Drawer
        open={open}
        onClose={toggleDrawer(false)}
        slotProps={{ paper: { style: { backgroundColor: '#524f4f' } } }}
      >
        {DrawerList}
      </Drawer>
    </div>
  );
}
